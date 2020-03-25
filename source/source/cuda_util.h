#ifndef _CUDA_UTIL_H_
#define _CUDA_UTIL_H_
#define THREAD_BLOCKS 512

#ifndef USE_CUDA
#define __host__
#define __global__
#define __device__
#define START 0
#define BLOCKID 1
#define STRIDE 1
inline void __syncthreads() {}
template <typename scalar>
inline scalar atomicAdd(scalar* address, scalar val)
{
    scalar old = *address;
    *address += val;
    return old;
}
template <typename scalar>
inline scalar atomicExch(scalar* address, scalar val)
{
    scalar old = *address;
    *address = val;
    return old;
}
template <typename scalar>
inline scalar atomicMax(scalar* address, scalar val)
{
    scalar old = *address;
    *address = (old < val ? val : old);
    return old;
}
template <typename scalar>
inline scalar atomicCAS(scalar* address, scalar compare, scalar val)
{
    scalar old = *address;
    *address = (old == compare ? val : old);
    return old;
}
struct Lock {
    void lock(void) {}
    void unlock(void) {}
};
#else
#include <cuda.h>
#include <cuda_runtime.h>
#define START (threadIdx.x)
#define BLOCKID (blockIdx.x)
#define STRIDE (blockDim.x)
struct Lock { // only for block
    int* mutex;
    Lock(int count)
    {
        int state = 0;
        cudaMalloc((void**)&mutex, count * sizeof(int));
        for (int i = 0; i < count; i++)
            cudaMemcpy(mutex + i, &state, sizeof(int), cudaMemcpyHostToDevice);
    }
    void Free()
    {
        cudaFree(mutex);
    }
    __device__ void lock(int x)
    {
        while ((int)atomicCAS(mutex+x, 0, 1) != 0)
            ;
    }
    __device__ void unlock(int x)
    {
        atomicExch(mutex+x, 0);
    }
};
#endif // USE_CUDA

#endif // _CUDA_UTIL_H_
