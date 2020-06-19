/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.matrix.word;

import seqtagsim.matrix.common;
import seqtagsim.util;

import cachetools.containers : HashMap;

import mir.ndslice;
import mir.math.common;
import mir.math.sum;

import std.stdio;
import std.typecons;
import std.range;
import std.datetime.stopwatch;
import std.experimental.allocator;
import std.experimental.allocator.showcase : mmapRegionList;
import std.experimental.allocator.mallocator;

/**
 * Vocabulary-based text overlap approach to create a label mapping in form a contigency table.
 */
struct Vocabulary
{
    alias Label = Tuple!(uint, "id", uint, "count");

    /**
     * Reads a range yielding tuples of word and tag.
     *
     * Params:
     *     range = InputRange yielding tuple of word and tag
     */
    void read(Range)(Range range) if (isInputRange!Range)
    {
        if (tokenCount == 0)
        {
            tags = makeArray!uint(Mallocator.instance, initialSize);
            tokens = makeArray!string(Mallocator.instance, initialSize);
        }
        foreach (string word, string tag; range)
        {
            Label* l = tag in labels;
            if (!l)
                l = labels.put(copy(tag), Label(labelCount++, 0));
            l.count++;
            tokens[tokenCount] = copy(word);
            tags[tokenCount++] = l.id;
            if (tokens.length == tokenCount)
            {
                expandArray(Mallocator.instance, tokens, tokens.length);
                expandArray(Mallocator.instance, tags, tags.length);
            }
        }
    }

    /**
     * Notify that the reading phase has ended. The read data will now be pre-processed.
     */
    void endReading()
    {
        shrinkArray(Mallocator.instance, tags, tags.length - tokenCount);
        shrinkArray(Mallocator.instance, tokens, tokens.length - tokenCount);
        tagList = makeArray!string(allocator, labels.length);
        labelCounts = makeArray!uint(allocator, labels.length);
        foreach (string text, Label info; labels.byPair)
        {
            tagList[info.id - 1] = text;
            labelCounts[info.id - 1] = info.count;
        }

        immutable positions = tagList.length + 1;
        foreach (string word, uint tag; lockstep(tokens, tags))
        {
            uint[]* fields = word in vocab;
            if (fields is null)
                fields = vocab.put(word, allocator.makeArray!uint(positions));
            (*fields)[0]++;
            (*fields)[tag]++;
        }
    }

    /**
     * Notify that the embedding phase begins. Does nothing as no embeddings are used.
     */
    void beginEmbedding()
    {
    }

    /**
     * Compares this Vocabulary with another Vocabulary.
     *
     * Params:
     *     other = other Vocabulary
     *     matchViaEmbedding = optional callback perform word matching via embedding if no direct match exists
     */
    auto compare(bool useEmbedding = false, V:
            Vocabulary)(ref V other, bool delegate(ref V, const size_t, ref uint[]*, ref double) matchViaEmbedding = null)
    {

        auto matrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto weightedMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto additiveMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto weightedAdditiveMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);

        // as above but will be multiplied by inverse label frequency
        auto ilfMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto ilfWeightedMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto ilfAdditiveMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);
        auto ilfWeightedAdditiveMatrix = rcslice!double([tagList.length, other.tagList.length].staticArray, double.epsilon);

        Progress progress = Progress(vocab.length);

        static if (useEmbedding)
        {
            import std.algorithm : copy;
            import core.atomic : atomicOp;
            import std.parallelism : taskPool;

            Atomic!ulong sharedVocabsCount;
            Atomic!ulong sharedWordsCountThis;
            Atomic!ulong sharedWordsCountOther;
            alias add = addAtomic;
            auto buf = allocator.makeArray!(Tuple!(string, uint[]))(vocab.length);
            const remaining = copy(vocab.byPair, buf);
            assert(remaining.length == 0);
            auto list = taskPool.parallel(buf);
        }
        else
        {
            ulong sharedVocabsCount;
            ulong sharedWordsCountThis;
            ulong sharedWordsCountOther;
            alias add = addPlain;
            auto list = enumerate(vocab.byPair);
        }

        foreach (size_t idx, pair; list)
        {
            progress++;
            const string word = pair[0];
            const uint[] values = pair[1];
            double similarity = 1.0;
            uint[]* o = word in other.vocab;
            if (o)
            {
                sharedVocabsCount += 1;
                sharedWordsCountThis += values[0];
                sharedWordsCountOther += (*o)[0];
            }
            else static if (!useEmbedding)
                continue;
            else if (!matchViaEmbedding(other, idx, o, similarity))
                continue;

            const uint[] otherValues = *o;
            foreach (tagId, count; values[1 .. $])
                if (count != 0)
                {
                    immutable double weight = count / cast(double) values[0];
                    immutable double labelWeight = 1.0 / labelCounts[tagId];
                    foreach (otherTagId, otherCount; otherValues[1 .. $])
                    {
                        immutable double otherLabelWeight = 1.0 / other.labelCounts[otherTagId];
                        immutable double otherWeight = otherCount / cast(double) otherValues[0];
                        add(matrix[tagId, otherTagId], count * otherCount * similarity); // bad
                        add(ilfMatrix[tagId, otherTagId], count * otherCount * labelWeight * otherLabelWeight * similarity); // bad
                        add(weightedMatrix[tagId, otherTagId], otherWeight * weight * similarity); // great
                        add(ilfWeightedMatrix[tagId, otherTagId], otherWeight * weight * labelWeight * otherLabelWeight * similarity); // great
                        add(additiveMatrix[tagId, otherTagId], (count + otherCount) * similarity); // awful
                        add(ilfAdditiveMatrix[tagId, otherTagId], (count * labelWeight + otherCount * otherLabelWeight) * similarity); // awful
                        add(weightedAdditiveMatrix[tagId, otherTagId], (count * otherWeight + otherCount * weight) * similarity); // great
                        add(ilfWeightedAdditiveMatrix[tagId, otherTagId],
                                (count * otherWeight * labelWeight + otherCount * weight * otherLabelWeight) * similarity); // great
                    }
                }
        }
        stderr.writeln("Filling matrix counts took ", progress.peek.total!"msecs", " ms");

        double sharedVocabularyFractionThis = sharedVocabsCount / cast(double) vocab.length;
        double sharedWordFractionThis = sharedWordsCountThis / cast(double) tokenCount;
        writefln!"Tokens A: %s / shared: %.3f"(tokenCount, sharedWordFractionThis);
        writefln!"Words A: %s / shared: %.3f"(vocab.length, sharedVocabularyFractionThis);

        double sharedVocabularyFractionOther = sharedVocabsCount / cast(double) other.vocab.length;
        double sharedWordFractionOther = sharedWordsCountOther / cast(double) other.tokenCount;
        writefln!"Tokens B: %s / shared: %.3f"(other.tokenCount, sharedWordFractionOther);
        writefln!"Words B: %s / shared: %.3f"(other.vocab.length, sharedVocabularyFractionOther);

        immutable ulong allTokensCount = tokenCount + other.tokenCount;
        ulong allWordsCount = vocab.length;
        ulong sharedTokensCount, sharedWordsCount;

        foreach (const string word, const uint[] counts; vocab.byPair)
        {
            const uint[]* otherCounts = word in other.vocab;
            if (otherCounts)
            {
                sharedTokensCount += (*otherCounts)[0] + counts[0];
                sharedWordsCount++;
            }
        }
        foreach (const string word, const uint[] counts; other.vocab.byPair)
            allWordsCount += word !in vocab;

        double sharedTokensFraction = sharedTokensCount / cast(double) allTokensCount;
        double sharedVocabularyFraction = sharedWordsCount / cast(double) allWordsCount;

        return tuple!("mul", "mulIwf", "mulIlf", "mulIwfIlf", "add", "addIwf", "addIlf", "addIwfIlf", "sharedTokens", "sharedVocabulary")(
                matrix, weightedMatrix, ilfMatrix, ilfWeightedMatrix,
                additiveMatrix, weightedAdditiveMatrix,
                ilfAdditiveMatrix, ilfWeightedAdditiveMatrix, sharedTokensFraction, sharedVocabularyFraction);
    }

private:
    enum initialSize = 1024;
    HashMap!(string, Label, Mallocator, false) labels;
    HashMap!(string, uint[], Mallocator, false) vocab;
    typeof(mmapRegionList(0)) allocator = mmapRegionList(4 * 1024 * 1024);
    string[] tagList;
    uint[] tags;
    string[] tokens;
    uint[] labelCounts;
    uint labelCount = 1;
    ulong tokenCount;

    pragma(inline, true) static void addAtomic(ref double current, const double value)
    {
        import core.atomic : atomicOp;

        // only used on allocated storage from matrices that are implicitly shared 
        atomicOp!"+="(*cast(shared double*)&current, value);
    }

    pragma(inline, true) static void addPlain(ref double current, const double value)
    {
        current += value;
    }

    string copy(const(char)[] s)
    {
        import std.exception : assumeUnique;

        return assumeUnique(makeArray(allocator, s));
    }
}

version (embedding)
{
    import seqtagsim.embedding;

    /**
     * Vocabulary-based text overlap with word embeddings for otherwise unmatched words.
     */
    @extends!Vocabulary struct EmbeddingTextOverlap(Embedding : EmbeddingBase)
    {
        mixin base;

        /**
         * Creates a new instance with the given embeddings.
         */
        this(ref Embedding emb, float similarityThreshold = 0.0f)
        {
            this.emb = &emb;
            this.similarityThreshold = similarityThreshold;
        }

        void beginEmbedding()
        {
            emb.beginEmbedding(vocab.length, true, (a, b) => writeln(a, "/", b));
            idToWord = allocator.makeArray!string(vocab.length);
            embeddings = allocator.makeUninitSlice!float(vocab.length, emb.embeddingDim);
            size_t idx;
            string[1] sentence;
            string[][1] sentences;
            sentences[0] = sentence;
            foreach (string word; vocab.byKey)
            {
                idToWord[idx] = word;
                sentence[0] = word;
                emb.embed(sentences[], embeddings[idx .. idx + 1]);
                idx++;
            }
            emb.endEmbedding();
        }

        auto compare(ref EmbeddingTextOverlap other)
        {
            return base.compare!true(other, &matchViaEmbedding);
        }

    private:
        Embedding* emb;
        string[] idToWord;
        Slice!(float*, 2) embeddings;
        float similarityThreshold;

        bool matchViaEmbedding(ref EmbeddingTextOverlap other, const size_t index, ref uint[]* o, ref double similarity)
        {
            immutable Tuple!(float, uint) maxIdx = computeSimilarity(other.embeddings, embeddings[index]);
            similarity = maxIdx[0];
            o = other.idToWord[maxIdx[1]] in other.vocab;
            return similarity > similarityThreshold;
        }

    }
}
