use std::thread;
use half::f16;
use crate::root::arch::safetensors::SafeTensor;
use crate::root::common_types::{Vecf16, Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::moe::ExpertLike;

pub struct LlamaExpert {
    pub gate_proj_w: WeightF16,
    pub up_proj_w: WeightF16,
    pub down_proj_w: WeightF16,
}

impl ExpertLike for LlamaExpert {
    fn gate_proj_w(&self) -> &WeightF16 { &self.gate_proj_w }
    fn up_proj_w(&self) -> &WeightF16 { &self.up_proj_w }
    fn down_proj_w(&self) -> &WeightF16 { &self.down_proj_w }
}

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

fn load_expert(safe_tensor: &mut SafeTensor, layer_no: usize, expert_no: usize, gpu: &mut Gpu) -> LlamaExpert {
    let gate_proj_w = load_expert_weight(safe_tensor, layer_no, expert_no, "gate_proj", gpu);
    let up_proj_w = load_expert_weight(safe_tensor, layer_no, expert_no, "up_proj", gpu);
    let down_proj_w = load_expert_weight(safe_tensor, layer_no, expert_no, "down_proj", gpu);

    LlamaExpert {gate_proj_w, up_proj_w, down_proj_w}
}

fn get_num_experts(safe_tensor: &mut SafeTensor, layer_no: usize) -> usize {
    let num_experts = (0..).take_while(|e| {
        safe_tensor.entries.contains_key(&format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_no, e))
    }).count();

    num_experts
}

fn read_and_convert(safe_tensor: &SafeTensor, layer_no: usize, expert_no: usize, weight_name: &str) -> Vecf16 {
    let full_name = format!("model.layers.{}.mlp.experts.{}.{}.weight", layer_no, expert_no, weight_name);
    let entry = &safe_tensor.entries[&full_name];
    let shard_idx = entry.shard_idx;
    let start = safe_tensor.data_offsets[shard_idx] as usize + entry.offsets[0] as usize;
    let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;
    let bytes = &safe_tensor.mmaps[shard_idx][start..start + byte_len];

    bytes.chunks_exact(2).map(|c| {
        let bits = u16::from_le_bytes(c.try_into().unwrap());
        let f32_val = f32::from_bits((bits as u32) << 16);
        f16::from_f32(f32_val)
    }).collect()
}
fn load_experts_raw_data(safe_tensor: &SafeTensor, num_experts: usize, layer_no: usize) -> Vec<(Vecf16, Vecf16, Vecf16)> {
    let raw_data: Vec<(Vecf16, Vecf16, Vecf16)> = thread::scope(|s| {
        let handles: Vec<_> = (0..num_experts).map(|e| {
            let st = &*safe_tensor;
            s.spawn(move || {
                let gate = read_and_convert(st, layer_no, e, "gate_proj");
                let up = read_and_convert(st, layer_no, e, "up_proj");
                let down = read_and_convert(st, layer_no, e, "down_proj");
                (gate, up, down)
            })
        }).collect();
        let res = handles.into_iter().map(|h| h.join().unwrap()).collect();
        res
    });

    raw_data
}

pub fn llama_load_experts(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &Gpu) -> Option<Vec<LlamaExpert>> {
    let num_experts = get_num_experts(safe_tensor, layer_no);
    let key = format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_no, 0);
    if safe_tensor.entries.contains_key(&key) {
        let raw_data: Vec<(Vecf16, Vecf16, Vecf16)> = load_experts_raw_data(safe_tensor, num_experts, layer_no );

        let mut exps = Vec::with_capacity(num_experts);
        for (gate, up, down) in &raw_data {
            let gate_proj_w = gpu.dev.htod_sync_copy(gate).unwrap();
            let up_proj_w = gpu.dev.htod_sync_copy(up).unwrap();
            let down_proj_w = gpu.dev.htod_sync_copy(down).unwrap();
            exps.push(LlamaExpert {gate_proj_w, up_proj_w, down_proj_w});
        }

        return Some(exps);
    }
    None
}


pub fn normalize_weights(weights: &mut Vecf32) {
    let sum = weights.iter().sum::<f32>();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}