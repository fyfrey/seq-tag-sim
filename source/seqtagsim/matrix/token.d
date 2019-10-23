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

import cachetools.containers : HashMap;
import mir.ndslice;
import mir.glas.l1;
import mir.glas.l2;

import seqtagsim.util;
import seqtagsim.embedding;
import seqtagsim.matrix.common;

/**
 * Token-based comparison of the entire dataset using contextual embeddings.
 */
struct Dataset(Embedding : EmbeddingBase)
{
    alias Label = Tuple!(uint, "id", uint, "count");

    this(ref Embedding emb, float similarityThreshold = 0.0)
    {
        this.emb = &emb;
        this.similarityThreshold = similarityThreshold;
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
        foreach (sentence; range.bySegment)
        {
            if (sentence.empty)
                continue;
            immutable tokensPriorSentence = tokens.length;
            foreach (string word, string tag; sentence)
            {
                Label* l = tag in labelMap;
                if (!l)
                    l = labelMap.put(copy(tag), Label(labelMap.length, 0));
                l.count++;
                tokens ~= copy(word);
                labels ~= cast(ubyte) l.id;
            }
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
        embeddings = makeUninitSlice!float(Mallocator.instance, tokens.length, emb.embeddingDim);
    }

    /**
     * Notify that the embedding phase begins. Embeds all tokens.
     */
    void beginEmbedding()
    {
        import std.algorithm : mapp = map, sum;

        size_t i;
        enum batchSize = 64;
        immutable numberOfBatches = (sentences.length + batchSize - 1) / batchSize;
        emb.beginEmbedding(numberOfBatches, (progress,
                total) => stderr.writef!"\rEmbedded %d of %d batches (%.1f%%)"(progress, total, 100 * progress / cast(double) total));
        string[] allTokens = tokens.data;
        foreach (uint[2][] batch; chunks(sentences.data, batchSize))
        {
            string[][batchSize] sentenceStorage;
            foreach (s, uint[2] indices; batch)
                sentenceStorage[s] = allTokens[indices[0] .. indices[1]];
            string[][] sentenceBatch = sentenceStorage[0 .. batch.length];
            immutable tokensInBatch = sentenceBatch.mapp!(s => s.length).sum;
            emb.normEmbeddings(sentenceBatch, embeddings[i .. i += tokensInBatch]);
        }
        emb.endEmbedding();
        stderr.writeln();
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

        size_t[2] dimensions = [labelMap.length, other.labelMap.length];
        size_t[2] dimensionsOther = [other.labelMap.length, labelMap.length];
        auto weightedMatrix = rcslice!double(dimensions, double.epsilon);
        auto matrix = rcslice!double(dimensions, double.epsilon);
        auto weightedMatrixOther = rcslice!double(dimensionsOther, double.epsilon);
        auto matrixOther = rcslice!double(dimensionsOther, double.epsilon);
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
            const ubyte[] thisLabels = labels.data;
            const ubyte[] otherLabels = other.labels.data;
            foreach (size_t i, Tuple!(float, uint) pair; batch)
            {
                immutable tagId = thisLabels[idx + i];
                immutable otherTagId = otherLabels[pair[1]];
                if (pair[0] > similarityThreshold)
                {
                    weightedMatrix[tagId, otherTagId] += pair[0];
                    matrix[tagId, otherTagId] += 1.0;
                }
                else
                    unmatchedTokenCountThis++;
            }
            progress += batch.length;
        };

        auto matrixFillCallbackOther = (Tuple!(float, uint)[] batch) {
            const ubyte[] thisLabels = labels.data;
            const ubyte[] otherLabels = other.labels.data;
            foreach (size_t i, Tuple!(float, uint) pair; batch)
            {
                immutable tagId = otherLabels[i];
                if (pair[1] >= thisLabels.length)
                {
                    writeln(pair, thisLabels.length);
                    return;
                }
                immutable otherTagId = thisLabels[pair[1]];
                if (pair[0] > similarityThreshold)
                {
                    weightedMatrixOther[tagId, otherTagId] += pair[0];
                    matrixOther[tagId, otherTagId] += 1.0;
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

        stderr.writeln("Filling matrix counts ms: ", progress.peek.total!"msecs");
        writefln!"Unmatched tokens: %d / %.1f %%"(unmatchedTokenCountThis, 100.0 * unmatchedTokenCountThis / labels.length);

        return tuple!("contextAB", "weightedAB", "contextBA", "weightedBA")(matrix, weightedMatrix, matrixOther, weightedMatrixOther);
    }

private:
    HashMap!(string, Label, Mallocator, false) labelMap;
    OutputBuffer!(ubyte, Mallocator) labels;
    Embedding* emb;
    Slice!(float*, 2, Contiguous) embeddings;
    float similarityThreshold = 0.0;
    typeof(mmapRegionList(0)) allocator = mmapRegionList(1024 * 1024);
    OutputBuffer!(string, Mallocator) tokens;
    OutputBuffer!(uint[2], Mallocator) sentences;

    string copy(const(char)[] s)
    {
        import std.exception : assumeUnique;

        return assumeUnique(makeArray(allocator, s));
    }

    void fallbackComputation(ref Progress progress, ref Slice!(float*, 2) thisEmb, ref Slice!(float*, 2) otherEmb,
            ubyte[] thisLabels, ubyte[] otherLabels, scope Slice!(double*, 2) matrix, scope Slice!(double*,
                2) weightedMatrix, ref ulong unmatchedTokenCount)
    {
        import std.parallelism : taskPool;
        import core.atomic : atomicOp;

        Atomic!ulong localUnmatchedTokenCount;
        foreach (size_t idx, ubyte tagId; taskPool.parallel(thisLabels))
        {
            immutable Tuple!(float, uint) maxIdx = computeSimilarity(otherEmb, thisEmb[idx]);
            immutable similarity = maxIdx[0];
            immutable otherTagId = otherLabels[maxIdx[1]];
            if (similarity > similarityThreshold)
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
}
