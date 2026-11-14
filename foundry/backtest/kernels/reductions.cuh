template<typename T> __device__ __forceinline__ T warp_reduce_sum(T v) {
    for (int off = warpSize >> 1; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

template<typename A>
__device__ __forceinline__ A warp_sum_bcast(A v) {
    v = warp_reduce_sum<A>(v);
    return __shfl_sync(0xffffffff, v, 0);
}
