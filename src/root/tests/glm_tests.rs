use crate::root::arch::quant::compressed_tensors::ct_dequantize;

pub fn glm_tests() {
    let packed: [i32; 1] = [0x75cfc582_u32 as i32];
    let scales: [f32; 1] = [0.00040435791015625];
    let out = ct_dequantize(&packed, &scales, 1, 8);
    println!("{:?}", out);
}