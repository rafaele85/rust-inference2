use std::thread;
use half::f16;
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Vecf16, Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::moe::StandardExpert;

fn load_expert_weight(safe_tensor: &mut SafeTensor, layer_no: usize, expert_no: usize, weight_name: &str, gpu: &mut Gpu) -> WeightF16 {
    let name = format!("mlp.experts.{}.{}", expert_no, weight_name);
    let full_name = format!("model.layers.{}.{}.weight", layer_no, name);

    let entry = &safe_tensor.entries[&full_name];
    let shard_idx = entry.shard_idx;
    let start = safe_tensor.data_offsets[shard_idx] as usize + entry.offsets[0] as usize;
    let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;
    let bytes = &safe_tensor.mmaps[shard_idx][start..start+byte_len];

    let w_gpu = gpu.upload_f16_from_bf16(bytes, &format!("{}.{}", layer_no, weight_name));

    w_gpu
}



pub fn normalize_weights(weights: &mut Vecf32) {
    let sum = weights.iter().sum::<f32>();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}