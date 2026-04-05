use crate::root::common_types::{Arrf32, Vecf32};

const CT_BITS: u32 = 4;
const CT_MASK: u32 = (1 << CT_BITS) - 1;
const CT_OFFSET: i32 = 8;
const CT_PACK_SIZE: usize = 32 / CT_BITS as usize;

fn get_packed<T>(packed: &[T], packed_cols: usize, row: usize, pc: usize) -> T
where  T: Copy {
    let packed_idx = row * packed_cols + pc;
    packed[packed_idx]
}
pub fn ct_dequantize(packed: &[i32], scales: &Arrf32, out_dim: usize, in_dim: usize) -> Vecf32 {
    let packed_cols = in_dim / CT_PACK_SIZE;
    let scale_cols = in_dim / 32;
    let mut output = vec![0.0f32; out_dim * in_dim];
    for row in 0..out_dim {
        for pc in 0..packed_cols {
            let val = get_packed(packed, packed_cols, row, pc) as u32;
            let scale = get_packed(scales, scale_cols, row, pc / 4);
            let offset = row * in_dim + pc * CT_PACK_SIZE;
            for k in 0..CT_PACK_SIZE {
                let shift = k as u32 * CT_BITS;
                let shifted_val: u32 = val >> shift;
                let nibble: u32 = shifted_val & CT_MASK;
                let diff = nibble as f32 - CT_PACK_SIZE as f32;
                let out_val = diff * scale;
                output[offset + k] = out_val;
            }
        }
    }

    output
}
