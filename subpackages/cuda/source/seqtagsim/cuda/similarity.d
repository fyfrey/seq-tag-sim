/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

module seqtagsim.cuda.similarity;

import std.algorithm : min, maxElement;
import std.typecons;
import std.stdio;
import std.range;
import std.parallelism;
import std.experimental.allocator;
import std.experimental.allocator.mallocator;

import mir.ndslice;

import seqtagsim.cuda.cublas;

/// Exception indicates an error occured during CUDA operations
final class CudaException : Exception
{
    @nogc @safe pure nothrow this(string msg, string file = __FILE__, size_t line = __LINE__, Throwable nextInChain = null)
    {
        super(msg, file, line, nextInChain);
    }
}

/// Finds the most similar indices according to their cosine similarity for two huge matrices in a batched manner 
void findMaxSimilarBatched(scope Slice!(const(float)*, 2) embeddings, scope Slice!(const(float)*, 2) otherEmbeddings,
        void delegate(size_t, Tuple!(float, uint)[]) callback, void delegate(Tuple!(float, uint)[]) callbackOther,
        const size_t maxBytes = size_t.max)
{
    try
    {
        CudaSimilarity cs = CudaSimilarity(otherEmbeddings, embeddings, maxBytes);
        Tuple!(float, uint)[] gpuIdx = Mallocator.instance.makeArray!(Tuple!(float, uint))(embeddings.length);
        Tuple!(float, uint)[] otherIdx = Mallocator.instance.makeArray!(Tuple!(float, uint))(otherEmbeddings.length);
        cudaHostRegister(gpuIdx.ptr, gpuIdx.length * gpuIdx[0].sizeof, cudaHostRegisterDefault);
        cudaHostRegister(otherIdx.ptr, otherIdx.length * otherIdx[0].sizeof, cudaHostRegisterDefault);
        scope (exit)
        {
            cudaHostUnregister(gpuIdx.ptr);
            cudaHostUnregister(otherIdx.ptr);
            Mallocator.instance.dispose(gpuIdx);
            Mallocator.instance.dispose(otherIdx);
        }
        cs.findMaxSimilarities(gpuIdx, callback, otherIdx, callbackOther);
    }
    catch (CudaException ce)
        stderr.writeln("Error during CUDA: '", ce.msg, "' in ", ce.file, ":", ce.line, "\nUsing fallback mechanism...");
}

private:

extern (C) void max_idx_all(const float* device_results, const uint rows, const uint columns, float* device_maxima,
        cudaStream_t stream = null);
extern (C) void max_idx_t(const float* device_results, const uint rows, const uint columns, float* device_maxima,
        const uint offset, const bool overwrite, cudaStream_t stream = null);

void checkError(int status, string file = __FILE__, int line = __LINE__)
{
    import std.string : fromStringz;
    import std.exception : assumeUnique;

    if (status != 0)
        throw new CudaException(assumeUnique(fromStringz(cudaGetErrorString(status))), file, line);
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
    }, (Tuple!(float, uint)[] batch) { count += batch.length; }, 1024 * 1024 * 725);
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
    }, 1024 * 1024 * 750);
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
    }, (Tuple!(float, uint)[] batch) { count += batch.length; }, 1024 * 1024 * 1024);
    assert(count == thisEmb.length + otherEmb.length);

    findMaxSimilarBatched(otherEmb, thisEmb, (size_t offset, Tuple!(float, uint)[] batch) { count -= batch.length; },
            (Tuple!(float, uint)[] allOther) {
        count -= allOther.length;
        assert(allOther.length == maxIdx.length);
        foreach (i, Tuple!(float, uint) o; allOther)
            if (maxIdx[i] != o)
            {
                writeln(i, ": ", maxIdx[i], " != ", o);
                if (count++ == 20)
                    assert(false);
            }
    }, 1024 * 1024 * 1024);
    assert(count == 0);
}

struct CudaSimilarity
{
    @disable this();
    @disable this(this);

    this(Slice!(const(float)*, 2) matrix, Slice!(const(float)*, 2) my, const size_t maxBytes = size_t.max)
    {
        length = matrix.shape[0];
        width = matrix.shape[1];
        vectorCount = my.shape[0];
        maxGpuMemory = maxBytes;
        initializeGpus(matrix, my);
    }

    ~this()
    {
        foreach (d; 0 .. deviceCount)
            cleanupGpuMemory(d);
    }

    void findMaxSimilarities(Tuple!(float, uint)[] maxIndex, void delegate(size_t, Tuple!(float, uint)[]) callback,
            Tuple!(float, uint)[] maxIndexOther, void delegate(Tuple!(float, uint)[]) callbackOther)
    {
        import core.thread : Thread, msecs;
        import std.range : transposed, enumerate, TransverseOptions;

        cudaEvent_t[maxGpuCount] finishEvents;
        Tuple!(float, uint)*[maxGpuCount] gpuMaxIdxOth;

        auto enqueue = (int device, Tuple!(float, uint)[] maxIndex) {
            cudaSetDevice(device).checkError;
            cudaEventCreateWithFlags(&(finishEvents[device]), cudaEventBlockingSync | cudaEventDisableTiming).checkError;
            immutable globalOffset = splitSizes[device][0];
            immutable globalEnd = splitSizes[device][1];
            size_t index;
            size_t bIdx;
            foreach (Tuple!(float, uint)[] batch; chunks(maxIndex[globalOffset .. globalEnd], batchSizes[device]))
            {
                while (finished[device] + 16 < bIdx)
                    Thread.sleep(10.msecs);
                events[device][bIdx++] = findMaxSimilaritiesBatch(device, globalOffset, index, batch);
                index += batch.length;
            }
            if (device == 0)
                gpuMaxIdxOth[0] = maxIndexOther.ptr;
            else
                cudaMallocHost(cast(void**)&gpuMaxIdxOth[device], maxIndexOther.length * maxIndexOther[0].sizeof).checkError;
            cudaMemcpyAsync(gpuMaxIdxOth[device], deviceIndexOther[device],
                    maxIndexOther.length * maxIndexOther[0].sizeof, cudaMemcpyDeviceToHost, stream[device]).checkError;
            cudaEventRecord(finishEvents[device], stream[device]).checkError;
        };

        typeof(scopedTask(enqueue, 0, maxIndex))[maxGpuCount] tasks;

        foreach (device; 0 .. deviceCount)
        {
            tasks[device] = scopedTask(enqueue, device, maxIndex);
            taskPool.put(tasks[device]);
        }

        scope (exit)
            foreach (device; 1 .. deviceCount)
                cudaFreeHost(gpuMaxIdxOth[device]).checkError;

        int retryCount;
        for (size_t progress; progress != maxIndex.length;)
            foreach (device; 0 .. deviceCount)
            {
                if (events[device].length == finished[device])
                    continue;
                cudaEvent_t batch = events[device][finished[device]];
                immutable status = cudaEventSynchronize(batch);
                if (status == cudaErrorInvalidResourceHandle || status == cudaErrorInvalidResourceHandleLegacy)
                {
                    if (retryCount++ == 1000)
                        checkError(status);
                    Thread.sleep(10.msecs);
                    continue;
                }
                retryCount = 0;
                checkError(status);
                cudaEventDestroy(batch);
                immutable deviceRange = splitSizes[device];
                immutable start = deviceRange[0] + finished[device] * batchSizes[device];
                immutable end = min(deviceRange[0] + ++finished[device] * batchSizes[device], deviceRange[1]);
                progress += end - start;
                callback(start, maxIndex[start .. end]);
            }
        foreach (ref task, ref event; lockstep(tasks[0 .. deviceCount], finishEvents[0 .. deviceCount]))
        {
            task.workForce();
            cudaEventSynchronize(event).checkError;
            cudaEventDestroy(event);
        }

        if (deviceCount > 1)
        {
            Tuple!(float, uint)[][maxGpuCount] maxIdxOth;
            foreach (i, ref arr; maxIdxOth[0 .. deviceCount])
                arr = gpuMaxIdxOth[i][0 .. maxIndexOther.length];
            foreach (i, pairs; transposed!(TransverseOptions.assumeNotJagged)(maxIdxOth[0 .. deviceCount]).enumerate)
                maxIndexOther[i] = maxElement!(a => a[0])(pairs);
        }

        callbackOther(maxIndexOther);
    }

private:
    enum maxGpuCount = 16;
    cublasHandle_t[maxGpuCount] handle;
    cudaStream_t[maxGpuCount] stream;
    float*[maxGpuCount] deviceMatrix;
    float*[maxGpuCount] deviceVector;
    float*[maxGpuCount] deviceResult;
    Tuple!(float, uint)*[maxGpuCount] deviceIndex;
    Tuple!(float, uint)*[maxGpuCount] deviceIndexOther;
    size_t length;
    size_t width;
    size_t[maxGpuCount] batchSizes;
    Tuple!(size_t, size_t)[maxGpuCount] splitSizes;
    size_t vectorCount;
    cudaEvent_t[][maxGpuCount] events;
    size_t[maxGpuCount] finished;
    int deviceCount;
    size_t maxGpuMemory = size_t.max;

    void initializeGpus(ref Slice!(const(float)*, 2) matrix, ref Slice!(const(float)*, 2) my)
    {
        cudaGetDeviceCount(&deviceCount).checkError;
        deviceCount = min(deviceCount, maxGpuCount);
        splitData();
        cudaHostRegister(cast(void*) matrix.ptr, matrix.elementCount * float.sizeof, cudaHostRegisterDefault).checkError;
        cudaHostRegister(cast(void*) my.ptr, my.elementCount * float.sizeof, cudaHostRegisterDefault).checkError;
        scope (exit)
        {
            cudaHostUnregister(cast(void*) my.ptr);
            cudaHostUnregister(cast(void*) matrix.ptr);
        }
        foreach (d; 0 .. deviceCount)
        {
            cudaSetDevice(d).checkError;
            allocateGpuMemory(d);
            cudaMemcpyAsync(deviceMatrix[d], matrix.ptr, matrix.elementCount * float.sizeof, cudaMemcpyHostToDevice, stream[d]).checkError;
            auto part = my[splitSizes[d][0] .. splitSizes[d][1]];
            cudaMemcpyAsync(deviceVector[d], part.ptr, part.elementCount * float.sizeof, cudaMemcpyHostToDevice, stream[d]).checkError;
        }
        foreach (d; 0 .. deviceCount)
        {
            cudaSetDevice(d).checkError;
            cudaDeviceSynchronize().checkError;
        }
    }

    void splitData()
    {
        import std.algorithm : stdMap = map, stdEach = each;
        import std.range : stdIota = iota;

        enumerate(evenChunks(stdIota(vectorCount), deviceCount).stdMap!(a => tuple(a[0], a[$ - 1] + 1))).stdEach!((i,
                r) => splitSizes[i] = r);
    }

    void allocateGpuMemory(int device)
    {
        cudaStreamCreateWithFlags(&(stream[device]), cudaStreamNonBlocking).checkError;
        cublasCreate_v2(&(handle[device])).checkError;
        cublasSetStream_v2(handle[device], stream[device]).checkError;

        immutable partSize = splitSizes[device][1] - splitSizes[device][0];
        batchSizes[device] = calculateBatchSize(device);
        immutable batches = (partSize + batchSizes[device] - 1) / batchSizes[device];
        stderr.writefln!"Preparing %d batches of size %d on GPU %d"(batches, batchSizes[device], device);
        events[device] = makeArray!cudaEvent_t(Mallocator.instance, batches);

        cudaMalloc(cast(void**)&(deviceResult[device]), batchSizes[device] * length * float.sizeof).checkError;
        cudaMalloc(cast(void**)&(deviceMatrix[device]), length * width * float.sizeof).checkError;
        cudaMalloc(cast(void**)&(deviceVector[device]), width * partSize * float.sizeof).checkError;
        cudaMalloc(cast(void**)&(deviceIndex[device]), partSize * Tuple!(float, uint).sizeof).checkError;
        cudaMalloc(cast(void**)&(deviceIndexOther[device]), length * Tuple!(float, uint).sizeof).checkError;
    }

    void cleanupGpuMemory(int device)
    {
        dispose(Mallocator.instance, events[device]);
        cudaFree(deviceMatrix[device]);
        cudaFree(deviceVector[device]);
        cudaFree(deviceResult[device]);
        cudaFree(deviceIndex[device]);
        cudaFree(deviceIndexOther[device]);
        cublasDestroy_v2(handle[device]);
        cudaStreamDestroy(stream[device]);
    }

    long calculateBatchSize(int device)
    {
        import std.algorithm : min;
        import std.format;

        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, device).checkError;
        immutable maxBlocks = properties.maxGridSize[0];

        enum memoryOverhead = 4 * 1024 * 1024;
        size_t free, total, mallocLimit;
        cudaMemGetInfo(&free, &total).checkError;
        cudaDeviceGetLimit(&mallocLimit, cudaLimitMallocHeapSize).checkError;
        if (free > mallocLimit)
        {
            immutable status = cudaDeviceSetLimit(cudaLimitMallocHeapSize, free);
            if (status != 0)
                free = mallocLimit;
        }
        free = min(maxGpuMemory, largestContiguousMemory(free));

        immutable partSize = splitSizes[device][1] - splitSizes[device][0];
        immutable matrixBytes = length * width * float.sizeof;
        immutable vectorsBytes = partSize * width * float.sizeof;
        immutable maxIndexBytes = partSize * Tuple!(float, uint).sizeof;
        immutable maxIndexOtherBytes = length * Tuple!(float, uint).sizeof;
        immutable required = matrixBytes + vectorsBytes + maxIndexBytes + memoryOverhead + maxIndexOtherBytes;
        if (free < required + length * float.sizeof)
        {
            throw new CudaException(format!"Unsufficient free memory on cuda device: Having only %d of required %d bytes."(free, required));
        }
        immutable available = free - required;
        immutable largestPossible = available / (length * float.sizeof);
        immutable warpMultiple = (largestPossible / properties.warpSize) * properties.warpSize;
        immutable finalBatchSize = min(warpMultiple, partSize, maxBlocks);
        return finalBatchSize;
    }

    size_t largestContiguousMemory(size_t freeBytes)
    {
        enum step = 1 << 22; // 4 MB
        enum cudaErrorMemoryAllocation = 2;
        void* buf;
        size_t retVal = (freeBytes / step) * step;
        while (cudaMalloc(&buf, retVal) == cudaErrorMemoryAllocation)
            retVal -= step;
        cudaFree(buf).checkError;
        return retVal;
    }

    cudaEvent_t findMaxSimilaritiesBatch(size_t device, size_t globalOffset, size_t idx, Tuple!(float, uint)[] maxIndex)
    {
        cudaEvent_t done;
        cudaEventCreateWithFlags(&done, cudaEventBlockingSync | cudaEventDisableTiming).checkError;
        immutable float alpha = 1.0f;
        immutable float beta = 0.0f;
        cublasSgemm_v2(handle[device], CUBLAS_OP_T, CUBLAS_OP_N, cast(int) length, cast(int) maxIndex.length,
                cast(int) width, &alpha, deviceMatrix[device], cast(int) width, deviceVector[device] + idx * width,
                cast(int) width, &beta, deviceResult[device], cast(int) length).checkError;
        max_idx_all(deviceResult[device], cast(uint) length, cast(uint) maxIndex.length,
                cast(float*)(deviceIndex[device] + idx), stream[device]);
        cudaMemcpyAsync(maxIndex.ptr, deviceIndex[device] + idx, maxIndex.length * maxIndex[0].sizeof,
                cudaMemcpyDeviceToHost, stream[device]).checkError;
        max_idx_t(deviceResult[device], cast(uint) length, cast(uint) maxIndex.length,
                cast(float*)(deviceIndexOther[device]), cast(uint)(globalOffset + idx), idx == 0, stream[device]);
        cudaEventRecord(done, stream[device]).checkError;
        return done;
    }
}

unittest
{
    import mir.random : Random;
    import mir.random.variable : uniformVar, normalVar;
    import mir.random.algorithm : randomSlice;

    enum length = 25049;
    enum width = 6020;

    float* matrix;
    Tuple!(float, uint)* maximaA;
    Tuple!(float, uint)* maximaB;

    cudaMalloc(cast(void**)&matrix, width * length * float.sizeof).checkError;
    cudaMalloc(cast(void**)&maximaA, length * maximaA[0].sizeof).checkError;
    cudaMalloc(cast(void**)&maximaB, width * maximaB[0].sizeof).checkError;
    cudaMemsetAsync(maximaB, 127, maximaB[0].sizeof * width).checkError;

    Random rng = Random(42);
    auto mat = randomSlice(rng, normalVar!float(0f, 0.5f), length, width);
    cudaMemcpy(matrix, mat.ptr, mat.elementCount * float.sizeof, cudaMemcpyHostToDevice).checkError;
    max_idx_all(matrix, width, length, cast(float*) maximaA);
    max_idx_t(matrix, width, length, cast(float*) maximaB, 0, true);
    cudaDeviceSynchronize().checkError;

    Tuple!(float, uint)[] maxA = new Tuple!(float, uint)[length];
    Tuple!(float, uint)[] maxB = new Tuple!(float, uint)[width];
    cudaMemcpy(maxA.ptr, maximaA, maxA.length * maxA[0].sizeof, cudaMemcpyDeviceToHost).checkError;
    cudaMemcpy(maxB.ptr, maximaB, maxB.length * maxB[0].sizeof, cudaMemcpyDeviceToHost).checkError;

    assert(mat.length == maxA.length);
    foreach (Slice!(float*) row, Tuple!(float, uint) res; lockstep(mat, maxA))
    {
        immutable size_t[1] index = maxIndex(row);
        assert(index[0] == res[1]);
        assert(row[index] == res[0]);
    }

    assert(mat.transposed.length == maxB.length);
    foreach (Slice!(float*, 1, Universal) col, Tuple!(float, uint) res; lockstep(mat.transposed, maxB))
    {
        immutable size_t[1] index = maxIndex(col);
        assert(index[0] == res[1]);
        assert(col[index] == res[0]);
    }
}
