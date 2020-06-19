/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.fasttext;

import std.experimental.allocator;

extern (C++)
{
private:
    struct Dstring
    {
        size_t length;
        const char* ptr;
    }

    struct Pair
    {
        float prob;
        Dstring word;
    }

    class FastTextWrapper
    {
        void* f;
        @disable this();
        final void loadModel(const char* filename, size_t length);

        final void fillWordVector(const char* word, size_t length, float* vector);

        final int getDimension();

        final void fillSentenceVector(const char* sentence, size_t length, float* vector);

        final void fillAnalogies(int k, const Dstring wordA, const Dstring wordB, const Dstring wordC, Pair*);

        final void fillNN(const Dstring word, int k, Pair* pairs);

        final void destroy();
    }

    Dstring copyString(Allocator)(void* alloc, const char* ptr, size_t length)
    {
        static if (__traits(compiles, (cast(Allocator*) alloc).makeArray!char(length)))
            char[] retVal = (cast(Allocator*) alloc).makeArray!char(length);
        else
            char[] retVal = (cast(shared Allocator*) alloc).makeArray!char(length);
        retVal[] = ptr[0 .. length];
        return *cast(Dstring*)&retVal;
    }

    Dstring copyString2(void* alloc, const char* ptr, size_t length)
    {
        return Dstring(0, null);
    }

    FastTextWrapper createInstance(void*, Dstring function(void* alloc, const char*, size_t) copyString);
}

/// D struct providing access to fastText with mir-based slice API
struct FastText(Allocator)
{
    import std.typecons : Tuple;
    import mir.ndslice.allocation : makeSlice;
    import mir.ndslice.slice : Slice, Contiguous;

    void loadModel(string filename)
    {
        f.loadModel(filename.ptr, filename.length);
    }

    Slice!(float*, 1, Contiguous) getWordVector(string word)
    {
        auto vector = makeSlice!float(alloc, f.getDimension());
        f.fillWordVector(word.ptr, word.length, vector.ptr);
        return vector;
    }

    void fillWordVector(string word, Slice!(float*, 1, Contiguous) vector)
    {
        assert(vector.length == getDimension());
        f.fillWordVector(word.ptr, word.length, vector.ptr);
    }

    void fillWordVector(string word, ref Slice!(float*, 1, Contiguous) vector)
    {
        assert(vector.length == getDimension());
        f.fillWordVector(word.ptr, word.length, vector.ptr);
    }

    float[] getSentenceVector(string sentence)
    {
        import std.algorithm : endsWith;

        if (!sentence.endsWith('\n'))
            sentence ~= '\n';
        float[] vector = alloc.makeArray!float(f.getDimension());
        f.fillSentenceVector(sentence.ptr, sentence.length, vector.ptr);
        return vector;
    }

    Tuple!(float, string)[] getAnalogies(int k, const string wordA, const string wordB, const string wordC)
    {
        Tuple!(float, string)[] retVal = alloc.makeArray!(Tuple!(float, string))(k);
        f.fillAnalogies(k, *cast(Dstring*)&wordA, *cast(Dstring*)&wordB, *cast(Dstring*)&wordC, cast(Pair*) retVal.ptr);
        return retVal;
    }

    Tuple!(float, string)[] getNN(const string word, int k)
    {
        Tuple!(float, string)[] retVal = alloc.makeArray!(Tuple!(float, string))(k);
        f.fillNN(*cast(Dstring*)&word, k, cast(Pair*) retVal.ptr);
        return retVal;
    }

    Tuple!(float, string) getNN(const string word)
    {
        Tuple!(float, string) retVal;
        f.fillNN(*cast(Dstring*)&word, 1, cast(Pair*)&retVal);
        return retVal;
    }

    size_t getDimension()
    {
        return f.getDimension();
    }

    this(ref Allocator alloc)
    {
        import std.traits : Unqual, CopyConstness;

        this.alloc = &alloc;
        f = createInstance(cast(void*) this.alloc, &copyString2);
    }

    @disable this(this);

    private Allocator* alloc;
    private FastTextWrapper f = void;
}
