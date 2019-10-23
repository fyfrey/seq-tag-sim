/**
 * Copyright: Fynn Schröder, 2019
 * Author: Fynn Schröder
 * License: MIT
 */

template<typename T, unsigned int blockSize>
__global__ 
void kernelMax(const T* __restrict__ input, 
        T* __restrict__ per_block_results, 
                const size_t n)
{
    __shared__ float sdata[blockSize];
    __shared__ unsigned int sidx[blockSize];

    T x = -1e38;
    unsigned int idx = UINT_MAX;
    for(unsigned int i=threadIdx.x; i < n; i += blockDim.x) {
        const T val = input[i + blockIdx.x * n];
        if (val > x)
        {
            x = val;
            idx = i;
        }
    }

    sdata[threadIdx.x] = x;
    sidx[threadIdx.x] = idx;
    __syncthreads();

    for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (threadIdx.x < s)
        {
            if (sdata[threadIdx.x + s] > sdata[threadIdx.x])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // thread 0 writes the final result
    if(threadIdx.x == 0) {
        per_block_results[blockIdx.x * 2] = sdata[0];
        per_block_results[blockIdx.x * 2 + 1] = __int_as_float(sidx[0]);
    }
}

extern "C" void max_idx_all(const float * device_results, const unsigned int rows, const unsigned int columns, float * device_maxima, cudaStream_t stream = NULL)
{
    dim3 blocksize(256); 
    dim3 gridsize(columns);
    kernelMax<float, 256><<<gridsize, blocksize, 0, stream>>>(device_results, device_maxima, rows);
}

template<typename T>
__global__ 
void kernelMaxOpt(const T* __restrict__ input, 
        T* __restrict__ per_block_results, 
                const size_t n, const size_t m, const unsigned int offset, const bool overwrite)
{
    T x = -1e38;
    unsigned int idx = UINT_MAX;

    for (int i = 0; i < n; i += 1)
    {
        if (threadIdx.x + blockIdx.x * blockDim.x < m)
        {
            float val = input[i * m + threadIdx.x + blockIdx.x * blockDim.x];
            if (val > x)
            {
                x = val;
                idx = i;
            }
        }
    }

    if (threadIdx.x + blockIdx.x * blockDim.x < m) {
        if (overwrite || x > per_block_results[(blockIdx.x * blockDim.x + threadIdx.x) * 2]) {
            per_block_results[(blockIdx.x * blockDim.x + threadIdx.x) * 2] = x;
            per_block_results[(blockIdx.x * blockDim.x + threadIdx.x) * 2 + 1] = __int_as_float(idx + offset);
        }
    }
}

extern "C" void max_idx_t(const float * device_results, const unsigned int rows, const unsigned int columns, float * device_maxima, const unsigned int offset, const bool overwrite, cudaStream_t stream = NULL)
{
    int blocksize = 256; 
    int gridsize = (rows + blocksize -1) / blocksize;
    kernelMaxOpt<float><<<gridsize, blocksize, 0, stream>>>(device_results, device_maxima, columns, rows , offset, overwrite);
}