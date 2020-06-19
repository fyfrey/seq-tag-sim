/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.matrix.token;

version (embedding):
import std.typecons;
import std.stdio;
import std.range : chunks, isInputRange;
import std.array;
import std.datetime.stopwatch;
import std.traits;
import std.experimental.allocator;
import std.experimental.allocator.showcase;
import std.experimental.allocator.mallocator;

import core.memory;

import cachetools.containers : HashMap;
import mir.ndslice;
import mir.glas.l1;
import mir.glas.l2;
import mir.math;

import seqtagsim.util;
import seqtagsim.embedding;
import seqtagsim.matrix.common;

/**
 * Token-based comparison of the entire dataset using contextual embeddings.
 */
struct Dataset(Embedding : EmbeddingBase)
{
    alias Label = Tuple!(uint, "id", uint, "count");

    /**
     * Creates a new instance.
     *
     * Params:
     *     emb = Contextual embeddings to use
     *     config = Configuration options
     */
    this(ref Embedding emb, const DatasetConfig config = DatasetConfig.init)
    {
        this.emb = &emb;
        this.config = config;
    }

    ~this()
    {
        if (embeddings != embeddings.init)
            Mallocator.instance.deallocate(embeddings.field);
        if (fuseCounts != fuseCounts.init)
            Mallocator.instance.deallocate(fuseCounts.field);
    }

    @disable this(this);

    @disable this();

    /**
     * Reads a range yielding segmenst that yield tuples of word and tag.
     *
     * Params:
     *     range = InputRange yielding another InputRange yielding tuples of word and tag
     */
    void read(Range)(Range range) if (isInputRange!Range && isInputRange!(ReturnType!((Range r) => r.bySegment)))
    {
        import std.algorithm.searching : find;

        foreach (sentence; range.bySegment)
        {
            if (sentence.empty)
                continue;
            auto tokensPriorSentence = tokens.length;
            foreach (string word, string tag; sentence)
            {
                Label* l = tag in labelMap;
                if (!l)
                {
                    if (config.fuseMultiTokenSpans)
                    {
                        uint encodedLabel = uniqueLabels | ((tag[0] == config.iMarker) << bitMaskPosition);
                        auto match = labelMap.byPair.find!((a, b) => a[0][1 .. $] == b)(tag[1 .. $]);
                        if (!match.empty)
                            encodedLabel = match.front[1].id ^ iMask;
                        else
                            uniqueLabels++;
                        l = labelMap.put(copy(tag), Label(encodedLabel, 0));
                    }
                    else
                        l = labelMap.put(copy(tag), Label(uniqueLabels++, 0));
                }
                l.count++;
                tokens ~= copy(word);
                labels ~= l.id;
                if (config.splitSentences && word.length == 1 && (word[0] == '.' || word[0] == '!' || word[0] == '?'))
                {
                    sentences ~= [cast(uint) tokensPriorSentence, cast(uint) tokens.length].staticArray;
                    tokensPriorSentence = tokens.length;
                }
            }
            if (tokensPriorSentence != tokens.length)
                sentences ~= [cast(uint) tokensPriorSentence, cast(uint) tokens.length].staticArray;
        }
    }

    /**
     * Notify that the reading phase has ended. The read data will now be pre-processed.
     */
    void endReading()
    {
        labels.minimize();
        tokens.minimize();
        sentences.minimize();
    }

    /**
     * Notify that the embedding phase begins. Embeds all tokens.
     */
    void beginEmbedding()
    {
        import std.algorithm : mapp = map, sum;

        embeddings = makeUninitSlice!float(Mallocator.instance, tokens.length, emb.embeddingDim);
        static immutable progressMessage = "\rEmbedded %d of %d batches (%.1f%%)";
        size_t i;
        enum batchSize = 64;
        immutable numberOfBatches = (sentences.length + batchSize - 1) / batchSize;
        immutable bool normalize = !config.fuseMultiTokenSpans;
        stderr.writef!progressMessage(0, numberOfBatches, 0.0);
        emb.beginEmbedding(numberOfBatches, normalize, (progress, total) => stderr.writef!progressMessage(progress,
                total, 100 * progress / cast(double) total));
        string[] allTokens = tokens.data;
        foreach (uint[2][] batch; chunks(sentences.data, batchSize))
        {
            string[][batchSize] sentenceStorage;
            foreach (s, uint[2] indices; batch)
                sentenceStorage[s] = allTokens[indices[0] .. indices[1]];
            string[][] sentenceBatch = sentenceStorage[0 .. batch.length];
            immutable tokensInBatch = sentenceBatch.mapp!(s => s.length).sum;
            emb.embed(sentenceBatch, embeddings[i .. i += tokensInBatch]);
        }
        emb.endEmbedding();
        stderr.writeln();
        fuseCounts = makeSlice!float(Mallocator.instance, [labels.length].staticArray, 1f);
        if (config.fuseMultiTokenSpans)
        {
            StopWatch sw = StopWatch(AutoStart.yes);
            immutable individualLabels = labels.length;
            stderr.writef!"Fusing %s individual tokens..."(individualLabels);
            fuseMultiTokenEmbeddings(0, labels.length);
            stderr.writefln!"done in %s ms! Fused %s multi-token spans -> %s final tokens for comparison."(sw.peek.total!"msecs",
                    individualLabels - labels.length, labels.length);
        }
    }

    /**
     * Compares this Dataset with another Dataset.
     *
     * Params:
     *     other = other Dataset
     */
    auto compare(ref Dataset other)
    {
        // assert no NaN values
        assert(embeddings.field.all!(x => x == x));
        assert(other.embeddings.field.all!(x => x == x));

        size_t[2] dimensions = [uniqueLabels, other.uniqueLabels];
        size_t[2] dimensionsOther = [other.uniqueLabels, uniqueLabels];
        auto weightedMatrix = rcslice!double(dimensions, double.epsilon);
        auto fusedWeightedMatrix = rcslice!double(dimensions, double.epsilon);
        auto matrix = rcslice!double(dimensions, double.epsilon);
        auto fusedMatrix = rcslice!double(dimensions, double.epsilon);
        auto weightedMatrixOther = rcslice!double(dimensionsOther, double.epsilon);
        auto fusedWeightedMatrixOther = rcslice!double(dimensionsOther, double.epsilon);
        auto matrixOther = rcslice!double(dimensionsOther, double.epsilon);
        auto fusedMatrixOther = rcslice!double(dimensionsOther, double.epsilon);
        immutable size_t lastPercent = cast(size_t)(labels.length * 0.01);
        Progress progress = Progress(labels.length + lastPercent);
        ulong unmatchedTokenCountThis;
        ulong unmatchedTokenCountOther;

        auto resetIfNecessary = () {
            if (progress.isComplete)
                return;
            progress.reset();
            matrix[] = double.epsilon;
            weightedMatrix[] = double.epsilon;
            unmatchedTokenCountThis = 0;
            unmatchedTokenCountOther = 0;
        };

        auto matrixFillCallback = (size_t idx, Tuple!(float, uint)[] batch) {
            const uint[] thisLabels = labels.data;
            const uint[] otherLabels = other.labels.data;
            foreach (size_t i, Tuple!(float, uint) pair; batch)
            {
                immutable tagId = thisLabels[idx + i];
                immutable otherTagId = otherLabels[pair[1]];
                immutable fuseFactor = fuseCounts[idx + i];
                immutable otherFuseFactor = other.fuseCounts[pair[1]];
                if (pair[0] > config.similarityThreshold)
                {
                    weightedMatrix[tagId, otherTagId] += pair[0];
                    matrix[tagId, otherTagId] += 1.0;
                    fusedMatrix[tagId, otherTagId] += fuseFactor * otherFuseFactor;
                    fusedWeightedMatrix[tagId, otherTagId] += pair[0] * fuseFactor * otherFuseFactor;
                }
                else
                    unmatchedTokenCountThis++;
            }
            progress += batch.length;
        };

        auto matrixFillCallbackOther = (Tuple!(float, uint)[] batch) {
            const uint[] thisLabels = labels.data;
            const uint[] otherLabels = other.labels.data;
            foreach (size_t i, Tuple!(float, uint) pair; batch)
            {
                immutable tagId = otherLabels[i];
                immutable fuseFactor = other.fuseCounts[i];
                if ((pair[1] >= thisLabels.length) | (pair[1] >= fuseCounts.length))
                {
                    stderr.writefln!"Error during computation at %s th token: %s %s"(i, pair, thisLabels.length);
                    continue;
                }
                immutable otherTagId = thisLabels[pair[1]];
                immutable otherFuseFactor = fuseCounts[pair[1]];
                if (pair[0] > config.similarityThreshold)
                {
                    weightedMatrixOther[tagId, otherTagId] += pair[0];
                    matrixOther[tagId, otherTagId] += 1.0;
                    fusedWeightedMatrixOther[tagId, otherTagId] += pair[0] * fuseFactor * otherFuseFactor;
                    fusedMatrixOther[tagId, otherTagId] += fuseFactor * otherFuseFactor;
                }
                else
                    unmatchedTokenCountOther++;
            }
            progress += lastPercent;
        };

        version (cuda)
            if (!progress.isComplete)
            {
                import seqtagsim.cuda.similarity : findMaxSimilarBatched;

                findMaxSimilarBatched(embeddings, other.embeddings, matrixFillCallback, matrixFillCallbackOther);
                resetIfNecessary();
            }

        version (blas)
            if (!progress.isComplete)
            {
                import seqtagsim.blas : findMaxSimilarBatched;

                findMaxSimilarBatched(embeddings, other.embeddings, matrixFillCallback, matrixFillCallbackOther);
                resetIfNecessary();
            }

        if (!progress.isComplete)
        {
            progress = Progress(labels.length + other.labels.length);
            stderr.writeln("Neither CUDA nor BLAS could be used, falling back to slower comparison.");
            fallbackComputation(progress, embeddings, other.embeddings, labels.data, other.labels.data,
                    matrix.lightScope, weightedMatrix.lightScope, unmatchedTokenCountThis);
            fallbackComputation(progress, other.embeddings, embeddings, other.labels.data, labels.data,
                    matrixOther.lightScope, weightedMatrixOther.lightScope, unmatchedTokenCountOther);
        }

        stderr.writeln("Filling matrix counts took ", progress.peek.total!"msecs", " ms");
        writefln!"Unmatched tokens: %d / %.1f %%"(unmatchedTokenCountThis, 100.0 * unmatchedTokenCountThis / labels.length);

        return tuple!("contextAB", "weightedAB", "contextBA", "weightedBA", "fusedAB", "fusedWeightedAB", "fusedBA", "fusedWeightedBA")(
                matrix, weightedMatrix,
                matrixOther, weightedMatrixOther, fusedMatrix, fusedWeightedMatrix, fusedMatrixOther, fusedWeightedMatrixOther);
    }

private:
    HashMap!(string, Label, Mallocator, false) labelMap;
    OutputBuffer!(uint, Mallocator) labels;
    Embedding* emb;
    Slice!(float*, 2) embeddings;
    Slice!(float*) fuseCounts;
    typeof(mmapRegionList(0)) allocator = mmapRegionList(1024 * 1024);
    OutputBuffer!(string, Mallocator) tokens;
    OutputBuffer!(uint[2], Mallocator) sentences;
    uint uniqueLabels;
    immutable DatasetConfig config;
    enum uint bitMaskPosition = 7;
    enum uint iMask = 1 << bitMaskPosition;

    string copy(const(char)[] s)
    {
        import std.exception : assumeUnique;

        return assumeUnique(makeArray(allocator, s));
    }

    void fallbackComputation(ref Progress progress, ref Slice!(float*, 2) thisEmb, ref Slice!(float*, 2) otherEmb,
            uint[] thisLabels, uint[] otherLabels, scope Slice!(double*, 2) matrix, scope Slice!(double*,
                2) weightedMatrix, ref ulong unmatchedTokenCount)
    {
        import std.parallelism : taskPool;
        import core.atomic : atomicOp;

        Atomic!ulong localUnmatchedTokenCount;
        foreach (size_t idx, uint tagId; taskPool.parallel(thisLabels))
        {
            immutable Tuple!(float, uint) maxIdx = computeSimilarity(otherEmb, thisEmb[idx]);
            immutable similarity = maxIdx[0];
            immutable otherTagId = otherLabels[maxIdx[1]];
            if (similarity > config.similarityThreshold)
            {
                atomicOp!"+="(*cast(shared double*)&weightedMatrix[tagId, otherTagId], similarity);
                atomicOp!"+="(*cast(shared double*)&matrix[tagId, otherTagId], 1.0);
            }
            else
                localUnmatchedTokenCount++;
            progress++;
        }
        unmatchedTokenCount = localUnmatchedTokenCount;
    }

    /**
     * Fuses same-label tokens together. Spans across function calls are not handled!
     */
    void fuseMultiTokenEmbeddings(size_t start, size_t end)
    {
        size_t removed;

        for (size_t i = start, current = start; i < end; i++, current++)
        {
            if ((labels[i] & iMask) == 0)
            {
                immutable uint activeTag = labels[i];
                immutable size_t spanStart = i;
                size_t spanEnd = spanStart + 1;
                for (; spanEnd < end && (labels[spanEnd] & iMask) && (labels[spanEnd] & ~iMask) == activeTag; spanEnd++)
                {
                }
                if (spanEnd != spanStart + 1)
                {
                    labels[current] = labels[spanStart];
                    embeddings[current][] = embeddings[spanStart];
                    foreach (Slice!(float*) e; embeddings[spanStart + 1 .. spanEnd])
                        embeddings[current][] += e;
                    embeddings[current][] *= 1f / nrm2(embeddings[current]);
                    fuseCounts[current] = spanEnd - spanStart;
                    i += spanEnd - spanStart - 1;
                    removed += spanEnd - spanStart - 1;
                    continue;
                }
            }
            labels[current] = labels[i];
            embeddings[current][] = embeddings[i] * (1f / nrm2(embeddings[i]));
        }
        labels.remove(labels.length - removed, labels.length);
        labels.minimize();
        embeddings = embeddings[0 .. $ - removed];
        embeddings._iterator = cast(embeddings.DeepElement*) pureRealloc(embeddings.ptr,
                embeddings.DeepElement.sizeof * embeddings.elementCount);
        fuseCounts = fuseCounts[0 .. $ - removed];
        fuseCounts._iterator = cast(fuseCounts.DeepElement*) pureRealloc(fuseCounts.ptr,
                fuseCounts.DeepElement.sizeof * fuseCounts.elementCount);
    }
}

unittest
{
    import std.range : enumerate;

    EmbeddingBase emb;
    Dataset!EmbeddingBase ds = Dataset!EmbeddingBase(emb);
    ds.labels ~= 0;
    ds.labels ~= 1;
    ds.labels ~= 2;
    ds.labels ~= 2 | ds.iMask;
    ds.labels ~= 2 | ds.iMask;
    ds.labels ~= 0;
    ds.labels ~= 1;
    ds.labels ~= 1 | ds.iMask;
    ds.labels ~= 2;
    ds.labels ~= 2 | ds.iMask;
    ds.labels ~= 1;
    ds.labels ~= 0;
    ds.embeddings = makeUninitSlice!float(Mallocator.instance, 12, 2);
    ds.embeddings[] = 0f;
    foreach (i, e; ds.embeddings.enumerate)
    {
        e[0] = i;
        e[1] = i * i;
        // e[] *= 1f / nrm2(e);
    }
    ds.fuseMultiTokenEmbeddings(0, 12);
    assert(ds.labels.data == [0, 1, 2, 0, 1, 2, 1, 0]);
    assert(ds.embeddings.length == ds.labels.length);
}

/// Configuration options for the token-based approach
struct DatasetConfig
{
    bool splitSentences = false;
    float similarityThreshold = 0.0f;
    bool fuseMultiTokenSpans = false;
    char iMarker = 'I';
}
