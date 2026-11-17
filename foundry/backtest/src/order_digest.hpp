/* Incremental SHA-256 of bounded chronological account behavior. */
#pragma once
#ifdef __CUDACC__
#define DIGEST_FN __device__ __forceinline__
#define DIGEST_ROUTINE_FN __device__ __noinline__
#else
#define DIGEST_FN inline
#define DIGEST_ROUTINE_FN inline
#endif
namespace foundry_order_digest {
enum { WORDS=26 };
DIGEST_FN unsigned int rotate(unsigned int x,int n) {return (x>>n)|(x<<(32-n));}
// Keep compression and word ingestion outside the inlined account branches.
DIGEST_ROUTINE_FN void block(unsigned int* state) {
    const unsigned int constants[64]={
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    unsigned int w[64];for(int i=0;i<16;++i)w[i]=state[10+i];
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for(int i=16;i<64;++i) {
        const unsigned int x=w[i-15],y=w[i-2];
        w[i]=w[i-16]+(rotate(x,7)^rotate(x,18)^(x>>3))+w[i-7]+(rotate(y,17)^rotate(y,19)^(y>>10));
    }
    unsigned int a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for(int i=0;i<64;++i) {
        const unsigned int t=h+(rotate(e,6)^rotate(e,11)^rotate(e,25))+((e&f)^(~e&g))+constants[i]+w[i];
        const unsigned int u=(rotate(a,2)^rotate(a,13)^rotate(a,22))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
    for(int i=10;i<26;++i)state[i]=0;
}
DIGEST_FN void initialize(unsigned int* state) {
    const unsigned int initial[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for(int i=0;i<WORDS;++i)state[i]=i<8?initial[i]:0;
}
DIGEST_FN void byte(unsigned int* state,unsigned int value) {
    const unsigned int position=state[8]&63;
    state[10+position/4]|=(value&255)<<(24-8*(position&3));
    if(++state[8]==0)++state[9];
    if((state[8]&63)==0)block(state);
}
DIGEST_ROUTINE_FN void integer(unsigned int* state,unsigned long long value) {
#ifdef __CUDACC__
#pragma unroll 1
#endif
    for(int i=7;i>=0;--i)byte(state,unsigned(value>>(i*8)));
}
DIGEST_FN void number(unsigned int* state,double value) {
    union {double value;unsigned long long bits;} encoded;
    encoded.value=value==0.?0.:value;integer(state,encoded.bits);
}
DIGEST_FN void finish(unsigned int* state) {
    const unsigned long long count=((unsigned long long)state[9]<<32)|state[8];
    byte(state,128);while((state[8]&63)!=56)byte(state,0);
    integer(state,count*8);
}
}
#undef DIGEST_FN
#undef DIGEST_ROUTINE_FN
