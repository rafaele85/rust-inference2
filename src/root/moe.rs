use crate::root::activation::{silu, softmax};
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::linear::{add_to_vector, hadamard, scale_and_add, ScaleAndAddArgs};
use crate::root::matmul::matmul_transposed_f16;

pub struct StandardExpert {
    pub gate_proj_w: WeightF16,
    pub up_proj_w: WeightF16,
    pub down_proj_w: WeightF16,
}


pub trait ExpertLike {
    fn gate_proj_w(&self) -> &WeightF16;
    fn up_proj_w(&self) -> &WeightF16;
    fn down_proj_w(&self) -> &WeightF16;
}


impl ExpertLike for StandardExpert {
    fn gate_proj_w(&self) -> &WeightF16 { &self.gate_proj_w }
    fn up_proj_w(&self) -> &WeightF16 { &self.up_proj_w }
    fn down_proj_w(&self) -> &WeightF16 { &self.down_proj_w }
}


pub struct MoeFFN<'a, TExpert: ExpertLike> {
    pub hidden: &'a mut Vecf32,
    pub normed: &'a Vecf32,
    pub router_w: &'a Vecf32,
    pub experts: &'a Vec<TExpert>,
    pub gpu: &'a mut Gpu,
    pub layer_no: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub hidden_size: usize,
}

pub fn moe_ffn<TExpert: ExpertLike>(args: MoeFFN<TExpert>) {
    let MoeFFN {mut hidden, normed, router_w, experts, gpu, num_experts_per_tok, moe_intermediate_size, hidden_size, .. } = args;
    let num_experts = experts.len();

    let scores = compute_scores(normed, router_w, hidden_size, num_experts);

    let (top_scores, top_indices) = get_top_scores(&scores, num_experts, num_experts_per_tok);

    let expert_probs  = softmax(&top_scores);

    let moe_out = apply_experts(ApplyExpertsArgs {
        top_indices: &top_indices, gpu, hidden_size, experts, normed, moe_intermediate_size, expert_probs : &expert_probs
    });

    add_to_vector(&mut hidden, &moe_out);
}



struct ApplyExpertsArgs<'a, TExpert: ExpertLike> {
    normed: &'a Vecf32,
    hidden_size: usize,
    top_indices: &'a Vec<usize>,
    experts: &'a Vec<TExpert>,
    moe_intermediate_size: usize,
    gpu: &'a mut Gpu,
    expert_probs : &'a Vecf32,
}

fn apply_experts<TExpert: ExpertLike>(args: ApplyExpertsArgs<TExpert>) -> Vecf32 {
    let ApplyExpertsArgs {hidden_size, top_indices, experts, normed, moe_intermediate_size, gpu, expert_probs} = args;
    let mut moe_out = vec![0.0f32; hidden_size];

    for (k, &expert_idx) in top_indices.iter().enumerate() {
        let expert = &experts[expert_idx];
        let w = expert_probs[k];
        apply_expert(ApplyExpertArgs {normed, hidden_size, w, moe_intermediate_size, moe_out: &mut moe_out, expert, gpu});
    }

    moe_out
}



struct ApplyExpertArgs<'a, TExpert: ExpertLike> {
    normed: &'a Vecf32,
    hidden_size: usize,
    expert: &'a TExpert,
    moe_intermediate_size: usize,
    w: f32,
    gpu: &'a mut Gpu,
    moe_out: &'a mut Vecf32,
}

fn apply_expert<TExpert: ExpertLike>(args: ApplyExpertArgs<TExpert>) {
    let ApplyExpertArgs {normed, expert, moe_intermediate_size,  gpu, hidden_size, w, mut moe_out} = args;
    let gate_proj_w = &expert.gate_proj_w();
    let up_proj_w = &expert.up_proj_w();
    let down_proj_w = &expert.down_proj_w();

    let gate = matmul_transposed_f16(normed, &gate_proj_w, moe_intermediate_size, gpu);
    let up = matmul_transposed_f16(normed, &up_proj_w, moe_intermediate_size, gpu);
    let silu_gate = silu(&gate);
    let ffn_act = hadamard(silu_gate, up, moe_intermediate_size);

    let expert_out = matmul_transposed_f16(&ffn_act, &down_proj_w, hidden_size, gpu);
    scale_and_add(ScaleAndAddArgs {output: &mut moe_out, scale: w, source: &expert_out, start_offset: 0, num_iterations: hidden_size});
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