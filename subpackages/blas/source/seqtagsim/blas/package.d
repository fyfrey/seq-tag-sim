/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.blas;

import std.typecons : Tuple, tuple;
import mir.ndslice;

/**
 * Finds the most similar pairwise embeddings.
 */
void findMaxSimilarBatched(scope Slice!(const(float)*, 2) embeddings, scope Slice!(const(float)*, 2) otherEmbeddings,
        void delegate(size_t, Tuple!(float, uint)[]) callback, void delegate(Tuple!(float, uint)[]) callbackOther,
        const size_t maxBytes = size_t.max)
{
    import mir.blas : gemm;
    import std.algorithm : min;
    import std.parallelism : taskPool;
    import std.experimental.allocator : makeArray, dispose;
    import std.experimental.allocator.mallocator : Mallocator;
    import std.stdio : stderr;

    immutable size_t thisLen = embeddings.length!0;
    immutable size_t otherLen = otherEmbeddings.length!0;

    Slice!(float*, 2) product = makeBatchStorage(thisLen, otherLen, maxBytes);
    scope (exit)
        destroyBatchStorage(product);
    immutable size_t batchSize = product.length!0;
    immutable size_t batches = (thisLen + batchSize - 1) / batchSize;
    size_t idx;

    Tuple!(float, uint)[] result = makeArray!(Tuple!(float, uint))(Mallocator.instance, batchSize);
    scope (exit)
        dispose(Mallocator.instance, result);
    Tuple!(float, uint)[] resultOther = makeArray!(Tuple!(float, uint))(Mallocator.instance, otherLen, tuple(-float.max, uint.max));
    scope (exit)
        dispose(Mallocator.instance, resultOther);

    stderr.writefln!"Computing %d batch(es) of size %d with BLAS"(batches, batchSize);

    foreach (b; 0 .. batches)
    {
        immutable size_t start = b * batchSize;
        immutable size_t end = min(start + batchSize, thisLen);
        immutable size_t len = end - start;
        Slice!(float*, 2) batchProduct = product[0 .. len];
        gemm(1f, embeddings[start .. end], otherEmbeddings.transposed, 0f, batchProduct);
        maxValueIndex(batchProduct, result);
        maxValueIndex(batchProduct.transposed, resultOther, cast(uint) start);
        callback(idx, result[0 .. len]);
        idx += len;
    }
    callbackOther(resultOther);
}

private:

unittest
{
    import mir.random : Random;
    import mir.random.variable : uniformVar;
    import mir.random.algorithm : randomSlice;
    import std.digest.sha : SHA1, toHexString;

    auto rng = Random(42);
    auto thisEmb = randomSlice(rng, uniformVar(0f, 1f), 6_789, 768);
    auto otherEmb = randomSlice(rng, uniformVar(0f, 1f), 34_567, 768);

    uint[] maxIdx = new uint[thisEmb.length!0];
    size_t count;

    findMaxSimilarBatched(thisEmb, otherEmb, (size_t offset, Tuple!(float, uint)[] batch) {
        count += batch.length;
        auto tmp = cast(ubyte[]) batch;
        assert(tmp.length == batch.length * 8);
        foreach (i, pair; batch)
            maxIdx[i + offset] = pair[1];
    }, (Tuple!(float, uint)[] allOther) { count += allOther.length; }, 1024 * 1024 * 64);
    assert(count == thisEmb.length + otherEmb.length);
}

unittest
{
    import mir.random : Random;
    import mir.random.variable : uniformVar;
    import mir.random.algorithm : randomSlice;
    import std.math : approxEqual;

    auto rng = Random(42);
    auto thisEmb = randomSlice(rng, uniformVar(0f, 1f), 100, 768);
    auto otherEmb = randomSlice(rng, uniformVar(0f, 1f), 234_567, 768);

    uint[] maxIdx = new uint[thisEmb.length!0];
    float[] maxSim = new float[thisEmb.length!0];
    size_t count;

    findMaxSimilarBatched(thisEmb, otherEmb, (size_t offset, Tuple!(float, uint)[] batch) {
        count += batch.length;
        foreach (i, pair; batch)
        {
            maxSim[i + offset] = pair[0];
            maxIdx[i + offset] = pair[1];
        }
    }, (Tuple!(float, uint)[] allOther) { count += allOther.length; }, 1024 * 1024 * 1024);
    assert(count == thisEmb.length + otherEmb.length);
    assert(maxIdx == [
            14890, 36070, 224363, 31719, 197796, 126598, 197796, 77858, 115305, 126598, 197796, 208388, 43719, 195371,
            177741, 168267, 31719, 86691, 208388, 209012, 84064, 71404, 197796, 31719, 108272, 84849, 86691, 209012,
            168267, 195371, 14890, 31044, 187781, 68340, 194462, 197796, 195371, 195371, 66917, 84064, 102054, 197796,
            197796, 168267, 53210, 197796, 142023, 46606, 208388, 209012, 23771, 194462, 208388, 197796, 23771, 86691,
            43719, 197796, 208388, 194462, 195371, 208388, 31719, 197796, 215690, 108272, 215690, 31719, 195371, 36070,
            84064, 169273, 121789, 197796, 66917, 197796, 189026, 66917, 86691, 84064, 53210, 6390, 149003, 218557,
            209012, 135166, 197796, 139853, 197796, 129410, 84064, 115305, 93136, 209012, 118930, 197796, 45013, 208388,
            66917, 173013
            ]);
    immutable float[] similarity = [
        211.635589599609375, 217.514373779296875, 209.2701416015625, 218.046295166015625, 210.504547119140625,
        215.3177490234375, 210.373382568359375, 201.800018310546875, 201.28387451171875, 203.1268310546875,
        216.761932373046875, 210.231689453125, 203.85369873046875, 218.679534912109375, 209.933013916015625,
        212.296630859375, 211.47100830078125, 219.433746337890625, 213.007659912109375, 222.012481689453125,
        203.0805511474609375, 218.222320556640625, 214.433837890625, 216.58331298828125, 211.373260498046875,
        212.6505279541015625, 213.0272064208984375, 209.2133941650390625, 203.947357177734375, 221.263275146484375,
        210.193084716796875, 218.889923095703125, 206.7576904296875, 204.035369873046875, 217.6998291015625,
        215.107330322265625, 217.07647705078125, 205.8101654052734375, 213.440948486328125, 210.40411376953125,
        209.382049560546875, 210.9676513671875, 214.697723388671875, 209.669097900390625, 215.8314208984375,
        206.623443603515625, 210.4749298095703125, 219.465850830078125, 221.32159423828125, 209.11541748046875,
        214.993408203125, 221.841278076171875, 215.18609619140625, 218.9025421142578125, 212.376434326171875,
        216.6827392578125, 218.4025421142578125, 217.3173675537109375, 218.5688934326171875, 211.680145263671875,
        210.539093017578125, 212.97515869140625, 209.889984130859375, 215.0991363525390625, 218.121490478515625,
        217.4267578125, 212.664398193359375, 212.20220947265625, 210.708221435546875, 211.7593994140625,
        212.95806884765625, 216.3385009765625, 211.1220703125, 219.9769134521484375, 213.255096435546875,
        208.65533447265625, 207.832000732421875, 215.5444183349609375, 214.8948516845703125, 214.69915771484375,
        213.646209716796875, 221.493927001953125, 216.614044189453125, 211.223724365234375, 215.4735107421875,
        210.233306884765625, 203.33367919921875, 210.462890625, 209.2219696044921875, 219.06182861328125,
        204.5558929443359375, 217.17962646484375, 207.821868896484375, 219.71343994140625, 210.8431854248046875,
        212.333709716796875, 215.4906768798828125, 218.21917724609375, 211.488861083984375, 207.64923095703125
    ];
    assert(maxSim.approxEqual(similarity));

    findMaxSimilarBatched(otherEmb, thisEmb, (size_t offset, Tuple!(float, uint)[] batch) { count -= batch.length; },
            (Tuple!(float, uint)[] allOther) {
        count -= allOther.length;
        assert(allOther.length == maxIdx.length);
        foreach (i, Tuple!(float, uint) o; allOther)
        {
            assert(maxSim[i].approxEqual(o[0]));
            assert(maxIdx[i] == o[1]);
        }
    }, 1024 * 1024 * 1024);
    assert(count == 0);
}

unittest
{
    import mir.random : Random;
    import mir.random.variable : uniformVar;
    import mir.random.algorithm : randomSlice;

    auto rng = Random(42);
    auto thisEmb = randomSlice(rng, uniformVar(0f, 1f), 25_049, 768);
    auto otherEmb = randomSlice(rng, uniformVar(0f, 1f), 25_006, 768);

    Tuple!(float, uint)[] maxIdx = new Tuple!(float, uint)[thisEmb.length];
    size_t count;

    findMaxSimilarBatched(thisEmb, otherEmb, (size_t offset, Tuple!(float, uint)[] batch) {
        count += batch.length;
        maxIdx[offset .. offset + batch.length] = batch;
    }, (Tuple!(float, uint)[] batch) { count += batch.length; }, 1024 * 1024 * 250);
    assert(count == thisEmb.length + otherEmb.length);

    findMaxSimilarBatched(otherEmb, thisEmb, (size_t offset, Tuple!(float, uint)[] batch) { count -= batch.length; },
            (Tuple!(float, uint)[] allOther) {
        count -= allOther.length;
        assert(allOther.length == maxIdx.length);
        foreach (i, Tuple!(float, uint) o; allOther)
            assert(maxIdx[i][1] == o[1]);
    }, 1024 * 1024 * 250);
    assert(count == 0);
}

import core.memory : pureFree, pureMalloc;

Slice!(float*, 2) makeBatchStorage(const size_t thisLen, const size_t otherLen, const size_t maxBytes = size_t.max)
{
    import std.algorithm : min, max;
    import std.array : staticArray;

    immutable minimalBatchSize = min(thisLen, 4);
    immutable rowSize = otherLen * float.sizeof;
    immutable availMem = getAvailableMemoryInBytes();
    size_t batchSize = min(thisLen, availMem.free / rowSize, availMem.total / (2 * rowSize), maxBytes / rowSize);
    void* mem = pureMalloc(batchSize * rowSize);
    while (!mem && batchSize > minimalBatchSize)
    {
        batchSize /= 2;
        mem = pureMalloc(batchSize * rowSize);
    }
    return sliced(cast(float*) mem, mem ? batchSize : 0, mem ? otherLen : 0);
}

void destroyBatchStorage(ref Slice!(float*, 2) storage)
{
    pureFree(storage.ptr);
}

Tuple!(size_t, "free", size_t, "total") getAvailableMemoryInBytes()
{
    enum unknown = tuple!("free", "total")(size_t.max, size_t.max);
    version (linux)
    {
        import core.sys.linux.sys.sysinfo : sysinfo, sysinfo_;

        sysinfo_ info;
        return sysinfo(&info) == 0 ? tuple!("free", "total")(info.freeram, info.totalram) : unknown;
    }
    else version (Windows)
    {
        import core.sys.windows.windows : MEMORYSTATUSEX, GlobalMemoryStatusEx;

        MEMORYSTATUSEX status = {status.sizeof};
        return GlobalMemoryStatusEx(&status) ? tuple!("free", "total")(status.ullAvailPhys, status.ullTotalPhys) : unknown;
    }
    else
    {
        return unknown;
    }
}

unittest
{
    Slice!(float*, 2) storage = makeBatchStorage(54_321, 12_345);
    scope (exit)
        destroyBatchStorage(storage);
    assert(storage.length!0 > 1);
    assert(storage.length!1 == 12_345);
}

void maxValueIndex(const Slice!(const(float)*, 2, Universal) sliceByColumns, ref Tuple!(float, uint)[] maxIdx, uint offset = 0)
{
    import std.algorithm : min;
    import std.range : iota;
    import std.parallelism : taskPool;

    enum uint rowBufSize = 1024 / float.sizeof;
    const Slice!(const(float)*, 2) slice = sliceByColumns.transposed.assumeContiguous;
    auto chunks = iota(0, slice.length!1, rowBufSize);

    foreach (c; taskPool.parallel(chunks))
    {
        float[rowBufSize] tempVal = void;
        uint[rowBufSize] tempIdx = void;
        tempVal[] = -float.max;
        immutable end = min(c + rowBufSize, slice.length!1);
        for (uint i = 0; i < slice.length; i++)
        {
            const Slice!(const(float)*, 1) rowPart = slice[i, c .. end];
            for (uint j = 0; j < rowPart.length; j++)
                if (rowPart[j] > tempVal[j])
                {
                    tempVal[j] = rowPart[j];
                    tempIdx[j] = i;
                }
        }
        for (uint i = 0; i + c < end; i++)
            if (tempVal[i] > maxIdx[i + c][0])
                maxIdx[i + c] = tuple(tempVal[i], tempIdx[i] + offset);
    }
}

void maxValueIndex(Slice!(const(float)*, 2) sliceByRows, ref Tuple!(float, uint)[] maxIdx)
{
    import std.parallelism : taskPool;

    foreach (i, Slice!(const(float)*) row; taskPool.parallel(sliceByRows))
    {
        uint index = uint.max;
        float value = -float.max;
        for (uint j = 0; j < row.length; j++)
            if (row[j] > value)
            {
                value = row[j];
                index = j;
            }
        maxIdx[i] = tuple(value, index);
    }
}

unittest
{
    import std.range : lockstep;
    import mir.random : Random;
    import mir.random.variable : uniformVar;
    import mir.random.algorithm : randomSlice;

    auto rng = Random(42);
    Slice!(float*, 2) orig = randomSlice(rng, uniformVar(0f, 1f), 50, 60);
    Slice!(float*, 2, Universal) trans = orig.transposed;
    Tuple!(float, uint)[] result = new Tuple!(float, uint)[trans.length];
    result[] = tuple(-float.max, uint.max);

    maxValueIndex(trans, result);

    foreach (Slice!(float*, 1, Universal) col, Tuple!(float, uint) res; lockstep(trans, result))
    {
        immutable size_t[1] index = maxIndex(col);
        assert(index[0] == res[1]);
        assert(col[index] == res[0]);
    }
}
