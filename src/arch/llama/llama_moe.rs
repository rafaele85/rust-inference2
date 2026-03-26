use crate::activation::softmax;
use crate::arch::llama::llama_activation::silu;
use crate::arch::llama::llama_config::LlamaConfig;
use crate::arch::llama::llama_linear::hadamard;
use crate::arch::llama::llama_load_weight::llama_load_weight;
use crate::arch::safetensors::SafeTensor;
use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;
use crate::linear::{add_to_vector, scale_and_add, ScaleAndAddArgs};
use crate::matmul::matmul_transposed;

pub struct LlamaExpert {
    pub gate_proj_w: Vecf32,
    pub up_proj_w: Vecf32,
    pub down_proj_w: Vecf32,
}

fn load_expert_weight(safe_tensor: &mut SafeTensor, layer_no: usize, expert_no: usize, weight_name: &str, gpu: &Gpu) -> Vecf32 {
    let name = format!("mlp.experts.{}.{}", expert_no, weight_name);
    let w = llama_load_weight(safe_tensor, &name, layer_no);
    w
}

fn load_expert(safe_tensor: &mut SafeTensor, layer_no: usize, expert_no: usize, gpu: &Gpu) -> LlamaExpert {
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

pub fn llama_load_experts(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &Gpu) -> Option<Vec<LlamaExpert>> {
    let num_experts = get_num_experts(safe_tensor, layer_no);
    let key = format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_no, 0);
    if safe_tensor.entries.contains_key(&key) {
        let mut exps = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let expert = load_expert(safe_tensor, layer_no, e, &gpu);
            exps.push(expert);
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
    for mut w in weights.iter_mut() {
        *w /= sum;
    }
}

struct ApplyExpertArgs<'a> {
    normed: &'a Vecf32,
    hidden_size: usize,
    expert: &'a LlamaExpert,
    moe_intermediate_size: usize,
    w: f32,
    gpu: &'a Gpu,
    moe_out: &'a mut Vecf32,
}

fn apply_expert(args: ApplyExpertArgs) {
    let ApplyExpertArgs {normed, expert, moe_intermediate_size,  gpu, hidden_size, w, mut moe_out} = args;
    let gate_proj_w = gpu.upload(&expert.gate_proj_w, "expert_gate_proj_w" );
    let up_proj_w = gpu.upload(&expert.up_proj_w, "expert_up_proj_w" );
    let down_proj_w = gpu.upload(&expert.down_proj_w, "expert_down_proj_w" );

    let gate = matmul_transposed(normed, &gate_proj_w, moe_intermediate_size, gpu);
    let up = matmul_transposed(normed, &up_proj_w, moe_intermediate_size, gpu);
    let silu_gate = silu(&gate);
    let ffn_act = hadamard(silu_gate, up, moe_intermediate_size);

    let expert_out = matmul_transposed(&ffn_act, &down_proj_w, hidden_size, gpu);
    scale_and_add(ScaleAndAddArgs {output: &mut moe_out, scale: w, source: &expert_out, start_offset: 0, num_iterations: hidden_size});
}
struct ApplyExpertsArgs<'a> {
    normed: &'a Vecf32,
    hidden_size: usize,
    top_indices: &'a Vec<usize>,
    experts: &'a Vec<LlamaExpert>,
    moe_intermediate_size: usize,
    gpu: &'a Gpu,
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
    pub gpu: &'a Gpu,
}

static mut MOE_DEBUG: bool = true;
pub fn llama_moe_ffn(args: LlamaMoeFFN) {
    let LlamaMoeFFN {mut hidden, normed, router_w, experts, config, gpu } = args;
    let LlamaConfig {num_experts_per_tok, moe_intermediate_size, hidden_size,norm_topk_prob, ..} = config;
    let num_experts = experts.len();

    let scores = compute_scores(normed, router_w, *hidden_size, num_experts);

    let (top_scores, top_indices) = get_top_scores(&scores, num_experts, *num_experts_per_tok);

    let mut expert_probs  = softmax(&top_scores);

    let debug = unsafe { MOE_DEBUG };
    if debug {
        unsafe { MOE_DEBUG = false; }
        println!("normed[:5]: {:?}", &normed[..5]);
        println!("router scores[:10]: {:?}", &scores[..10]);
        println!("top-8 indices: {:?}", &top_indices[..8]);
        println!("expert_probs: {:?}", &expert_probs);
        println!("router_w len: {}", router_w.len());
        println!("router_w[:10]: {:?}", &router_w[..10]);
    }

    if *norm_topk_prob {
        normalize_weights(&mut expert_probs);
    }

    let moe_out = apply_experts(ApplyExpertsArgs {
        top_indices: &top_indices, gpu, hidden_size: *hidden_size, experts, normed, moe_intermediate_size: *moe_intermediate_size, expert_probs : &expert_probs
    });

    add_to_vector(&mut hidden, &moe_out);
}
