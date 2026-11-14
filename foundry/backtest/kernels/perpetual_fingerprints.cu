extern "C" __global__ void perpetual_fingerprints(const unsigned long long* weights,
    unsigned long long* out, long long count) {
    __shared__ unsigned long long hashes[256];
    unsigned long long h = 0;
    for (long long j = threadIdx.x; j < count; j += blockDim.x) {
        const unsigned long long v = weights[(long long)blockIdx.x * count + j];
        unsigned long long x = (v << 1) == 0 ? 0ULL : v;
        x ^= (unsigned long long)(j + 1) * 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        h ^= x ^ (x >> 31);
    }
    hashes[threadIdx.x] = h;
    __syncthreads();
    for (int offset = 128; offset; offset >>= 1) {
        if (threadIdx.x < offset) hashes[threadIdx.x] ^= hashes[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = hashes[0];
}
