use crate::root::common_types::Vecf32;

pub fn mlx_affine_dequantize(
    packed: &[u32],   // quantized weights
    scales: &[f32],
    biases: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    bits: usize,
) -> Vecf32 {
    let pack_size = 32 / bits;  // 8 for 4-bit
    let packed_cols = in_dim / pack_size;
    let mask = (1u32 << bits) - 1;
    let scale_cols = in_dim / group_size;
    let mut output = vec![0.0f32; out_dim * in_dim];

    for row in 0..out_dim {
        for pc in 0..packed_cols {
            let val = packed[row * packed_cols + pc];
            let col_base = pc * pack_size;
            for k in 0..pack_size {
                let col = col_base + k;
                let group = col / group_size;
                let shift = k as u32 * bits as u32;
                let w_int = (val >> shift) & mask;
                let scale = scales[row * scale_cols + group];
                let bias = biases[row * scale_cols + group];
                output[row * in_dim + col] = w_int as f32 * scale + bias;
            }
        }
    }
    output
}