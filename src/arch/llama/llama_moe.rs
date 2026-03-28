use std::thread;
use half::f16;
use crate::activation::softmax;
use crate::arch::llama::llama_activation::silu;
use crate::arch::llama::llama_config::LlamaConfig;
use crate::arch::llama::llama_linear::hadamard;
use crate::arch::safetensors::SafeTensor;
use crate::common_types::{Vecf16, Vecf32, WeightF16};
use crate::gpu::Gpu;
use crate::linear::{add_to_vector, scale_and_add, ScaleAndAddArgs};
use crate::matmul::{matmul_transposed_f16};

pub struct LlamaExpert {
    pub gate_proj_w: WeightF16,
    pub up_proj_w: WeightF16,
    pub down_proj_w: WeightF16,
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

fn compute_scores(normed: &Vecf32, router_w: &Vecf32, hidden_size: usize, num_experts: usize) -> Vecf32 {
    let mut scores = vec![0.0f32; num_experts];
    for i in 0..num_experts {
        let mut dot = 0.0f32;
        for j in 0..hidden_size {
            dot += normed[j] * router_w[i * hidden_size + j];
        }
        scores[i] = dot;
    }
    scores
}

fn get_top_scores(scores: &Vecf32, num_experts: usize, num_experts_per_tok: usize) -> (Vecf32, Vec<usize>) {
    let mut top_indices: Vec<usize> = (0..num_experts).collect();
    top_indices.sort_by(|&a, &b| {
        let score_b = scores[b];
        let score_a = scores[a];
        let res = score_b.partial_cmp(&score_a).unwrap();
        res
    });
    top_indices.truncate(num_experts_per_tok);

    let top_scores: Vecf32 = top_indices.iter().map(|&i| scores[i]).collect();
    (top_scores, top_indices)
}

pub fn normalize_weights(weights: &mut Vecf32) {
    let sum = weights.iter().sum::<f32>();
    for w in weights.iter_mut() {
        *w /= sum;
    }
}

struct ApplyExpertArgs<'a> {
    normed: &'a Vecf32,
    hidden_size: usize,
    expert: &'a LlamaExpert,
    moe_intermediate_size: usize,
    w: f32,
    gpu: &'a mut Gpu,
    moe_out: &'a mut Vecf32,
}

fn apply_expert(args: ApplyExpertArgs) {
    let ApplyExpertArgs {normed, expert, moe_intermediate_size,  gpu, hidden_size, w, mut moe_out} = args;
    let gate_proj_w = &expert.gate_proj_w;
    let up_proj_w = &expert.up_proj_w;
    let down_proj_w = &expert.down_proj_w;

    let gate = matmul_transposed_f16(normed, &gate_proj_w, moe_intermediate_size, gpu);
    let up = matmul_transposed_f16(normed, &up_proj_w, moe_intermediate_size, gpu);
    let silu_gate = silu(&gate);
    let ffn_act = hadamard(silu_gate, up, moe_intermediate_size);

    let expert_out = matmul_transposed_f16(&ffn_act, &down_proj_w, hidden_size, gpu);
    scale_and_add(ScaleAndAddArgs {output: &mut moe_out, scale: w, source: &expert_out, start_offset: 0, num_iterations: hidden_size});
}
struct ApplyExpertsArgs<'a> {
    normed: &'a Vecf32,
    hidden_size: usize,
    top_indices: &'a Vec<usize>,
    experts: &'a Vec<LlamaExpert>,
    moe_intermediate_size: usize,
    gpu: &'a mut Gpu,
    expert_probs : &'a Vecf32,
}

fn apply_experts(args: ApplyExpertsArgs) -> Vecf32 {
    let ApplyExpertsArgs {hidden_size, top_indices, experts, normed, moe_intermediate_size, gpu, expert_probs} = args;
    let mut moe_out = vec![0.0f32; hidden_size];

    for (k, &expert_idx) in top_indices.iter().enumerate() {
        let expert = &experts[expert_idx];
        let w = expert_probs[k];
        apply_expert(ApplyExpertArgs {normed, hidden_size, w, moe_intermediate_size, moe_out: &mut moe_out, expert, gpu});
    }

    moe_out
}

pub struct LlamaMoeFFN<'a> {
    pub hidden: &'a mut Vecf32,
    pub normed: &'a Vecf32,
    pub router_w: &'a Vecf32,
    pub experts: &'a Vec<LlamaExpert>,
    pub config: &'a LlamaConfig,
    pub gpu: &'a mut Gpu,
    pub layer_no: usize,
}

pub fn llama_moe_ffn(args: LlamaMoeFFN) {
    let LlamaMoeFFN {mut hidden, normed, router_w, experts, config, gpu, .. } = args;
    let LlamaConfig {num_experts_per_tok, moe_intermediate_size, hidden_size, ..} = config;
    let num_experts = experts.len();

    let scores = compute_scores(normed, router_w, *hidden_size, num_experts);

    let (top_scores, top_indices) = get_top_scores(&scores, num_experts, *num_experts_per_tok);

    let expert_probs  = softmax(&top_scores);

    let moe_out = apply_experts(ApplyExpertsArgs {
        top_indices: &top_indices, gpu, hidden_size: *hidden_size, experts, normed, moe_intermediate_size: *moe_intermediate_size, expert_probs : &expert_probs
    });

    add_to_vector(&mut hidden, &moe_out);
}
