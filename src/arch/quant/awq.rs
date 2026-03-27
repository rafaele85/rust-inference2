use crate::common_types::{Arrf32, Arri32, Vecf32};

const AWQ_BITS: u32 = 4;
const AWQ_PACK_SIZE: usize = 32 / AWQ_BITS as usize; // 8
const AWQ_MASK: u32 = (1 << AWQ_BITS) - 1;           // 0xF
const AWQ_ORDER: [u32; AWQ_PACK_SIZE] = [0, 4, 1, 5, 2, 6, 3, 7];

pub struct AwqDequantizeArgs<'a> {
    pub qweight: &'a Arri32,
    pub qzero: &'a Arri32,
    pub scales: &'a Arrf32,
    pub in_dim: usize,
    pub out_dim: usize,
    pub group_size: usize,
}
pub fn awq_dequantize(args: AwqDequantizeArgs) -> Vecf32 {
    let AwqDequantizeArgs {qweight, qzero, scales, in_dim, out_dim, group_size} = args;
    let out_dim_packed = out_dim / 8;
    let mut output = vec![0.0f32; out_dim * in_dim];

    for in_idx in 0..in_dim {
        let group = in_idx / group_size;
        let z_offset = group * out_dim_packed;
        let w_offset = in_idx * out_dim_packed;

        for packed_col in 0..out_dim_packed {
            let packed_val = qweight[w_offset + packed_col] as u32;
            let z_packed = qzero[z_offset + packed_col] as u32;
            let col_offset = packed_col * AWQ_PACK_SIZE;
            let scale_offset = group * out_dim + col_offset;
            let out_offset = col_offset * in_dim + in_idx;
            let out_offset = col_offset + in_idx * out_dim;
            dequantlize_col(DequantizeColArgs {output: &mut output, packed_val, z_packed, scales, out_offset, in_dim, scale_offset});
        }
    }
    output
}

pub struct DequantizeColArgs<'a> {
    output: &'a mut Vecf32,
    packed_val: u32,
    z_packed: u32,
    scales: &'a Arrf32,
    out_offset: usize,
    in_dim: usize,
    scale_offset: usize,
}
fn dequantlize_col(args: DequantizeColArgs) {
    let DequantizeColArgs {output, packed_val, z_packed, scales, out_offset, in_dim, scale_offset} = args;

    for k in 0..AWQ_PACK_SIZE {
        let shift = AWQ_ORDER[k] * AWQ_BITS;

        let w_int4 = ((packed_val >> shift) & AWQ_MASK) as f32;
        let z_int4 = ((z_packed >> shift) & AWQ_MASK) as f32;
        let scale = scales[scale_offset + k];
        let out = (w_int4 - z_int4) * scale;

        output[out_offset + k] = out;
    }
}