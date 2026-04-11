use crate::root::activation::{sigmoid, silu, softmax};
use crate::root::arch::qwen35::qwen35_layer::Qwen35MoeWeights;
use crate::root::common_types::Vecf32;
use crate::root::config::MoeConfig;
use crate::root::gpu::{Gpu};
use crate::root::linear::{add_to_vector, hadamard, scale_and_add, sum_products, ScaleAndAddArgs};
use crate::root::matmul::matmul_transposed_f16;
use crate::root::moe::{compute_scores, get_top_scores};

pub fn qwen35_moe_ffn(
    hidden: &mut Vecf32,
    normed: &Vecf32,
    moe: &Qwen35MoeWeights,
    num_experts_per_tok: usize,
    gpu: &mut Gpu,
) {
    let hidden_size = moe.experts.hidden_size;
    let moe_intermediate = moe.experts.moe_intermediate;
    let num_experts = moe.experts.num_experts;
    let group_size = moe.experts.group_size;

    let scores = compute_scores(normed, &moe.router_w, hidden_size, num_experts);
    let all_probs = softmax(&scores);
    let (top_probs, top_indices) = get_top_scores(&all_probs, num_experts, num_experts_per_tok);
    let mut moe_out = vec![0.0f32; hidden_size];

    for (k, &expert_idx) in top_indices.iter().enumerate() {
        let w = top_probs[k];
        let experts = &moe.experts;

        let gate = experts.matmul(normed,
                                  &experts.gate_proj_packed, &experts.gate_proj_scales, &experts.gate_proj_biases,
                                  expert_idx, hidden_size, moe_intermediate, group_size, gpu);

        let up = experts.matmul(normed,
                                &experts.up_proj_packed, &experts.up_proj_scales, &experts.up_proj_biases,
                                expert_idx, hidden_size, moe_intermediate, group_size, gpu);

        let silu_gate = silu(&gate);
        let ffn_act = hadamard(silu_gate, up, moe_intermediate);

        let expert_out = experts.matmul(&ffn_act,
                                        &experts.down_proj_packed, &experts.down_proj_scales, &experts.down_proj_biases,
                                        expert_idx, moe_intermediate, hidden_size, group_size, gpu);

        scale_and_add(ScaleAndAddArgs { scale: w, output: &mut moe_out, start_offset: 0, num_iterations: hidden_size, source: &expert_out });
    }

    add_to_vector(hidden, &moe_out);
    qwen35_shared_expert(hidden, normed, moe, gpu);
}

fn qwen35_shared_expert(
    hidden: &mut Vecf32,
    normed: &Vecf32,
    moe: &Qwen35MoeWeights,
    gpu: &mut Gpu,
) {
    let experts = &moe.experts;
    let hidden_size = experts.hidden_size;
    let intermediate = experts.moe_intermediate;

    let gate_proj_w = &moe.shared_gate_proj_w;
    let up_proj_w = &moe.shared_up_proj_w;
    let down_proj_w = &moe.shared_down_proj_w;


    let gate = matmul_transposed_f16(normed, &gate_proj_w, intermediate, gpu);
    let up = matmul_transposed_f16(normed, &up_proj_w, intermediate, gpu);
    let silu_gate = silu(&gate);
    let ffn_act = hadamard(silu_gate, up, intermediate);
    let shared_out = matmul_transposed_f16(&ffn_act, &down_proj_w, hidden_size, gpu);

    let gate_score = sum_products(normed, &moe.shared_expert_gate_w, hidden_size);
    let gate_sigmoid = sigmoid(gate_score);
    scale_and_add(ScaleAndAddArgs {output: hidden, start_offset: 0, source: &shared_out, scale: gate_sigmoid, num_iterations: hidden_size});
}
