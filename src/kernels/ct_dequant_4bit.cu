#include <cuda_fp16.h>

extern "C" __global__ void ct_dequant_4bit(
    const unsigned int* packed,
    const __half* scales,
    __half* output,
    int out_dim,
    int packed_cols,
    int scale_cols,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_dim * packed_cols;
    if (idx >= total) return;

    int row = idx / packed_cols;
    int pc = idx % packed_cols;

    unsigned int val = packed[idx];

    for (int k = 0; k < 8; k++) {
        int col = pc * 8 + k;
        int group = col / group_size;
        int scale_idx = row * scale_cols + group;

        unsigned int nibble = (val >> (k * 4)) & 0xF;
        float s = __half2float(scales[scale_idx]);
        float result = ((float)nibble - 8.0f) * s;

        output[row * (packed_cols * 8) + col] = __float2half(result);
    }
}