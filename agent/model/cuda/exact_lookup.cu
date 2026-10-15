#include <stdint.h>
// Preserve the tested deterministic gather accumulation order: complete
// 32-value passes, descending shuffle tree, then sequential remainder.
extern "C" __global__ void exact_lookup_sum(const float* grad, float* output,
    const int64_t* unique, const int64_t* positions, const int64_t* counts,
    const int64_t* offsets, int64_t width) {
    int lane=threadIdx.x&31;
    int64_t feature=(int64_t)blockIdx.y*4+threadIdx.x/32;
    if(feature>=width)return;
    int64_t group=blockIdx.x, count=counts[group], start=offsets[group];
    float value=0.f;
    for(int64_t i=0;i<count/32;i++)
        value+=grad[positions[start+i*32+lane]*width+feature];
    for(int offset=16;offset;offset/=2)value+=__shfl_down_sync(0xffffffff,value,offset);
    if(lane==0){
        for(int64_t i=(count/32)*32;i<count;i++)
            value+=grad[positions[start+i]*width+feature];
        output[unique[group]*width+feature]=0.f+value;
    }
}
