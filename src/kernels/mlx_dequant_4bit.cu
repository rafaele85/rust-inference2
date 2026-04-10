#include <cuda_fp16.h>

extern "C" __global__ void mlx_dequant_4bit(
  const unsigned int* __restrict__ packed,
  const __half* __restrict__ scales,
  const __half* __restrict__ biases,
  __half* __restrict__ output,
  int out_dim,
  int packed_cols,
  int scale_cols,
  int group_size
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_packed = out_dim * packed_cols;
  if (idx >= total_packed) {
    return;
  }

  int row = idx / packed_cols;
  int pc = idx % packed_cols;

  unsigned int val = packed[idx];
  int col_base = pc * 8;
  int out_base = row * packed_cols * 8 + col_base;
  int scale_row = row * scale_cols;

  for (int k=0; k<8; k++) {
    int col = col_base + k;
    int group = col / group_size;
    int shift = k * 4;
    unsigned int nibble = (val >> shift) & 0XF;
    int scale_idx = scale_row + group;
    float s = __half2float(scales[scale_idx]);
    float b = __half2float(biases[scale_idx]);
    float result = (float)nibble * s + b;
    int out_idx = out_base + k;
    output[out_idx] = __float2half(result);
  }
}