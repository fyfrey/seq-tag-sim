/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 *
 * D translation of selected parts from cuda.h and https://docs.nvidia.com/cuda/archive/10.1/cuda-runtime-api/index.html
 */

module seqtagsim.cuda.cuda;

version (cuda) extern (C) @nogc nothrow:

alias cudaStream_t = void*;
alias cudaEvent_t = void*;

alias cudaError = int;
enum : cudaError
{
    cudaSuccess = 0,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorInvalidResourceHandleLegacy = 33,
    cudaErrorNotReady = 600
}
alias cudaError_t = cudaError;

alias cudaMemcpyKind = int;
enum : cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
}

enum cudaHostRegisterDefault = 0x00;
enum cudaStreamDefault = 0x00;
enum cudaStreamNonBlocking = 0x01;
enum cudaEventBlockingSync = 0x01;
enum cudaEventDisableTiming = 0x02;

alias cudaLimit = int;
enum cudaLimitMallocHeapSize = 0x02;

cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, uint flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = null);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream = null);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit);
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value);
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, uint flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = null);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaHostRegister(void* ptr, size_t size, uint flags);
cudaError_t cudaHostUnregister(void* ptr);
const(char)* cudaGetErrorString(cudaError_t error);

struct cudaDeviceProp
{
    char[256] name;
    char[16] uuid;
    char[8] luid;
    uint luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int[3] maxThreadsDim;
    int[3] maxGridSize;
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int[2] maxTexture2D;
    int[2] maxTexture2DMipmap;
    int[3] maxTexture2DLinear;
    int[2] maxTexture2DGather;
    int[3] maxTexture3D;
    int[3] maxTexture3DAlt;
    int maxTextureCubemap;
    int[2] maxTexture1DLayered;
    int[3] maxTexture2DLayered;
    int[2] maxTextureCubemapLayered;
    int maxSurface1D;
    int[2] maxSurface2D;
    int[3] maxSurface3D;
    int[2] maxSurface1DLayered;
    int[3] maxSurface2DLayered;
    int maxSurfaceCubemap;
    int[2] maxSurfaceCubemapLayered;
    size_t surfaceAlignment;
    int concurrentKernels;
    int eccEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
}
