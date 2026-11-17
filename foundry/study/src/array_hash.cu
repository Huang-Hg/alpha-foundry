// SHA-256 of independent 4096-byte leaves. Standard FIPS 180-4 compression.
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

__device__ __constant__ uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ __forceinline__ uint32_t rotr(uint32_t v, int n) { return (v >> n) | (v << (32-n)); }

extern "C" __global__ void sha256_leaves(const unsigned char* input, unsigned char* output,
                                       long long row_bytes, int leaves, int count) {
    int item = blockIdx.x * blockDim.x + threadIdx.x;
    if (item >= count * leaves) return;
    int row = item / leaves, leaf = item % leaves;
    long long offset = (long long)leaf * 4096;
    int size = (int)min(4096LL, row_bytes - offset);
    const unsigned char* data = input + (long long)row * row_bytes + offset;
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                     0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    int blocks = (size + 9 + 63) / 64;
    for (int block = 0; block < blocks; ++block) {
        uint32_t w[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint32_t word = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int at = block * 64 + i * 4 + j;
                unsigned char byte = 0;
                if (at < size) byte = data[at];
                else if (at == size) byte = 0x80;
                else if (at >= blocks * 64 - 8)
                    byte = (unsigned char)(((uint64_t)size * 8) >> ((blocks * 64 - 1 - at) * 8));
                word = (word << 8) | byte;
            }
            w[i] = word;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (int i = 0; i < 64; ++i) {
            if (i >= 16) {
                uint32_t x=w[(i-15)&15], y=w[(i-2)&15];
                w[i&15] += (rotr(x,7)^rotr(x,18)^(x>>3)) + w[(i-7)&15]
                           + (rotr(y,17)^rotr(y,19)^(y>>10));
            }
            uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            uint32_t t1=z+s1+((e&f)^((~e)&g))+K[i]+w[i&15];
            uint32_t t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&b)^(a&c)^(b&c));
            z=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=z;
    }
    #pragma unroll
    for (int i=0;i<8;++i) {
        #pragma unroll
        for (int j=0;j<4;++j) output[(long long)item*32+i*4+j]=(unsigned char)(h[i]>>(24-j*8));
    }
}
