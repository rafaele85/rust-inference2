use std::thread;
use half::f16;
use crate::root::activation::{silu, softmax};
use crate::root::arch::quant::compressed_tensors::ct_dequantize;
use crate::root::arch::safetensors::{load_scales, load_shape_dims, load_tensor_awq, load_tensor_i32, SafeTensor};
use crate::root::common_types::{Vecf16, Vecf32, WeightF16};
use crate::root::config::{ModelDimConfig, MoeConfig};
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


pub struct MoeFFN<'a, TExpert: ExpertLike, TConfig: MoeConfig + ModelDimConfig> {
    pub hidden: &'a mut Vecf32,
    pub normed: &'a Vecf32,
    pub router_w: &'a Vecf32,
    pub experts: &'a Vec<TExpert>,
    pub gpu: &'a mut Gpu,
    pub config: &'a TConfig,
}

pub fn moe_ffn<TExpert: ExpertLike, TConfig: MoeConfig + ModelDimConfig>(args: MoeFFN<TExpert, TConfig>) {
    let MoeFFN {
        mut hidden, normed, router_w, experts, gpu, config
    } = args;
    let num_experts = experts.len();

    let scores = compute_scores(normed, router_w, config.get_hidden_size(), num_experts);

    let all_probs = softmax(&scores);
    let (mut top_probs, top_indices) = get_top_scores(&all_probs, num_experts, config.get_num_experts_per_tok());

    if config.get_norm_topk_prob() {
        let sum: f32 = top_probs.iter().sum();
        if sum > 0.0 {
            for p in top_probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    let moe_out = apply_experts(ApplyExpertsArgs {
        top_indices: &top_indices, gpu, experts, normed, expert_probs : &top_probs, config
    });

    if config.get_routed_scaling_factor() != 1.0 {
        let scaled: Vecf32 = moe_out.iter().map(|&x| x * config.get_routed_scaling_factor()).collect();
        add_to_vector(&mut hidden, &scaled);
    } else {
        add_to_vector(&mut hidden, &moe_out);
    }


}



struct ApplyExpertsArgs<'a, TExpert: ExpertLike, TConfig: ModelDimConfig + MoeConfig> {
    normed: &'a Vecf32,
    top_indices: &'a Vec<usize>,
    experts: &'a Vec<TExpert>,
    gpu: &'a mut Gpu,
    expert_probs : &'a Vecf32,
    config: &'a TConfig,
}

fn apply_experts<TExpert: ExpertLike, TConfig: ModelDimConfig + MoeConfig>(args: ApplyExpertsArgs<TExpert, TConfig>) -> Vecf32 {
    let ApplyExpertsArgs {top_indices, experts, normed, config, gpu, expert_probs} = args;
    let mut moe_out = vec![0.0f32; config.get_hidden_size()];

    for (k, &expert_idx) in top_indices.iter().enumerate() {
        let expert = &experts[expert_idx];
        let w = expert_probs[k];
        apply_expert(ApplyExpertArgs {normed, w, config, moe_out: &mut moe_out, expert, gpu});
    }

    moe_out
}



struct ApplyExpertArgs<'a, TExpert: ExpertLike, TConfig: ModelDimConfig + MoeConfig> {
    normed: &'a Vecf32,
    expert: &'a TExpert,
    w: f32,
    gpu: &'a mut Gpu,
    moe_out: &'a mut Vecf32,
    config: &'a TConfig,
}

fn apply_expert<TExpert: ExpertLike, TConfig: ModelDimConfig + MoeConfig>(args: ApplyExpertArgs<TExpert, TConfig>) {
    let ApplyExpertArgs {normed, expert, gpu, w, mut moe_out, config} = args;
    let gate_proj_w = &expert.gate_proj_w();
    let up_proj_w = &expert.up_proj_w();
    let down_proj_w = &expert.down_proj_w();

    let gate = matmul_transposed_f16(normed, &gate_proj_w, config.get_moe_intermediate_size(), gpu);
    let up = matmul_transposed_f16(normed, &up_proj_w, config.get_moe_intermediate_size(), gpu);
    let silu_gate = silu(&gate);
    let ffn_act = hadamard(silu_gate, up, config.get_moe_intermediate_size());

    let expert_out = matmul_transposed_f16(&ffn_act, &down_proj_w, config.get_hidden_size(), gpu);
    scale_and_add(ScaleAndAddArgs {output: &mut moe_out, scale: w, source: &expert_out, start_offset: 0, num_iterations: config.get_hidden_size()});
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


fn get_num_experts(safe_tensor: &mut SafeTensor, layer_no: usize) -> usize {
    let num_experts = (0..).take_while(|e| {
        safe_tensor.entries.contains_key(&format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_no, e))
            || safe_tensor.entries.contains_key(&format!("model.layers.{}.mlp.experts.{}.gate_proj.qweight", layer_no, e))
            || safe_tensor.entries.contains_key(&format!("model.layers.{}.mlp.experts.{}.gate_proj.weight_packed", layer_no, e))
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


fn load_experts_raw_data_bf16(safe_tensor: &SafeTensor, num_experts: usize, layer_no: usize) -> Vec<(Vecf16, Vecf16, Vecf16)> {
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

pub fn load_experts_bf16(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &Gpu) -> Option<Vec<StandardExpert>> {
    let num_experts = get_num_experts(safe_tensor, layer_no);
    if num_experts == 0 {
        return None;
    }

    let raw_data: Vec<(Vecf16, Vecf16, Vecf16)> = load_experts_raw_data_bf16(safe_tensor, num_experts, layer_no);

    let mut exps = Vec::with_capacity(num_experts);
    for (gate, up, down) in &raw_data {
        let gate_proj_w = gpu.dev.htod_sync_copy(gate).unwrap();
        let up_proj_w = gpu.dev.htod_sync_copy(up).unwrap();
        let down_proj_w = gpu.dev.htod_sync_copy(down).unwrap();
        exps.push(StandardExpert { gate_proj_w, up_proj_w, down_proj_w });
    }
    Some(exps)
}

pub fn load_experts_awq(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> Option<Vec<StandardExpert>> {
    let num_experts = get_num_experts(safe_tensor, layer_no);
    if num_experts == 0 {
        return None;
    }

    let mut exps = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gate_f32 = load_tensor_awq(safe_tensor, &format!("model.layers.{}.mlp.experts.{}.gate_proj", layer_no, e));
        let up_f32 = load_tensor_awq(safe_tensor, &format!("model.layers.{}.mlp.experts.{}.up_proj", layer_no, e));
        let down_f32 = load_tensor_awq(safe_tensor, &format!("model.layers.{}.mlp.experts.{}.down_proj", layer_no, e));

        let gate_proj_w = gpu.upload_f16_from_f32(&gate_f32, &format!("exp{}gate{}", layer_no, e));
        let up_proj_w = gpu.upload_f16_from_f32(&up_f32, &format!("exp{}up{}", layer_no, e));
        let down_proj_w = gpu.upload_f16_from_f32(&down_f32, &format!("exp{}down{}", layer_no, e));

        exps.push(StandardExpert { gate_proj_w, up_proj_w, down_proj_w });
    }
    Some(exps)
}


fn load_experts_raw_data_ct(safe_tensor: &mut SafeTensor, num_experts: usize, layer_no: usize) -> Vec<(Vecf32, Vecf32, Vecf32)> {
    // Step 1: read all packed/scale/shape data sequentially (needs &mut)
    let mut raw_inputs: Vec<(Vec<i32>, Vecf32, usize, usize, Vec<i32>, Vecf32, usize, usize, Vec<i32>, Vecf32, usize, usize)> = Vec::new();

    for e in 0..num_experts {
        let prefix = format!("model.layers.{}", layer_no);
        let mut read_ct = |name: &str| -> (Vec<i32>, Vecf32, usize, usize) {
            let packed_key = format!("{}.mlp.experts.{}.{}.weight_packed", prefix, e, name);
            let scale_key = format!("{}.mlp.experts.{}.{}.weight_scale", prefix, e, name);
            let shape_key = format!("{}.mlp.experts.{}.{}.weight_shape", prefix, e, name);
            let packed = load_tensor_i32(safe_tensor, &packed_key);
            let scales = load_scales(safe_tensor, &scale_key);
            let (out_dim, in_dim) = load_shape_dims(safe_tensor, &shape_key);
            (packed, scales, out_dim, in_dim)
        };
        let (gp, gs, go, gi) = read_ct("gate_proj");
        let (up, us, uo, ui) = read_ct("up_proj");
        let (dp, ds, do_, di) = read_ct("down_proj");
        raw_inputs.push((gp, gs, go, gi, up, us, uo, ui, dp, ds, do_, di));
    }

    // Step 2: dequantize in parallel (pure computation, no &mut needed)
    thread::scope(|s| {
        let handles: Vec<_> = raw_inputs.iter().map(|input| {
            s.spawn(|| {
                let gate = ct_dequantize(&input.0, &input.1, input.2, input.3);
                let up = ct_dequantize(&input.4, &input.5, input.6, input.7);
                let down = ct_dequantize(&input.8, &input.9, input.10, input.11);
                (gate, up, down)
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

pub fn load_experts_ct(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu) -> Option<Vec<StandardExpert>> {
    let num_experts = get_num_experts(safe_tensor, layer_no);

    let raw_data = load_experts_raw_data_ct(safe_tensor, num_experts, layer_no);

    let mut exps = Vec::with_capacity(num_experts);
    for (gate, up, down) in &raw_data {
        let gate_proj_w = gpu.upload_f16_from_f32(gate, "exp_gate");
        let up_proj_w = gpu.upload_f16_from_f32(up, "exp_up");
        let down_proj_w = gpu.upload_f16_from_f32(down, "exp_down");
        exps.push(StandardExpert { gate_proj_w, up_proj_w, down_proj_w });
    }
    Some(exps)
}