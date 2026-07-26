use crate::root::activation::{sigmoid, silu};
use crate::root::arch::qwen35::qwen35_config::TextConfig;
use crate::root::arch::qwen35::qwen35_layer::Qwen35LinearAttnLayer;
use crate::root::common_types::{Arrf32, Vecf32};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::KVCache;
use crate::root::linear::{add_to_vector, sum_products_with_offset};
use crate::root::matmul::{matmul_transposed_f16};
use crate::root::norm::{l2_norm, rms_norm, rms_norm_zero_centered, softplus};

pub struct Qwen35LinearQKV {
    pub q: Vecf32,  // [num_k_heads * key_head_dim] = [2048]
    pub k: Vecf32,  // [num_k_heads * key_head_dim] = [2048]
    pub v: Vecf32,  // [num_v_heads * value_head_dim] = [8192]
}

pub struct Qwen35DeltaGates {
    pub decay: Vecf32,
    pub beta: Vecf32,
}

fn fill_conv_state(conv_state: &mut Vecf32, qkv_proj: &Vecf32, qkv_dim: usize, kernel_size: usize) {
    for ch in 0..qkv_dim {
        let base = ch * kernel_size;

        for k in 0..kernel_size - 1 {
            conv_state[base + k] = conv_state[base + k + 1];
        }
        conv_state[base + kernel_size - 1] = qkv_proj[ch];
    }
}

fn fill_output(qkv_dim: usize, kernel_size: usize, conv_state: &Vecf32, conv_weight: &Vecf32) -> Vecf32 {
    let mut output = vec![0.0f32; qkv_dim];

    for ch in 0..qkv_dim {
        let state_base = ch * kernel_size;
        let weight_base = ch * kernel_size;

        let sum = sum_products_with_offset(conv_state, state_base, conv_weight, weight_base, kernel_size);
        output[ch] = sum;
    }

    output
}
pub fn qwen35_conv1d_update(
    qkv_proj: &Vecf32,
    conv_state: &mut Vecf32,
    conv_weight: &Vecf32,
    qkv_dim: usize,
    kernel_size: usize,
) -> Vecf32 {
    if conv_state.is_empty() {
        *conv_state = vec![0.0f32; qkv_dim * kernel_size];
    }

    fill_conv_state(conv_state, qkv_proj, qkv_dim, kernel_size);


    let output = fill_output(qkv_dim, kernel_size, conv_state, conv_weight);

    output
}

pub fn qwen35_split_qkv(
    qkv: &Vecf32,
    tc: &TextConfig,
) -> Qwen35LinearQKV {
    let q_dim = tc.linear_num_key_heads * tc.linear_key_head_dim;
    let k_dim = q_dim;
    let v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim;

    let q_raw = &qkv[..q_dim];
    let k_raw = &qkv[q_dim..q_dim + k_dim];
    let v_raw = &qkv[q_dim + k_dim..q_dim + k_dim + v_dim];

    let q = l2_norm(q_raw, tc.linear_key_head_dim, tc.linear_num_key_heads, tc.rms_norm_eps);
    let k = l2_norm(k_raw, tc.linear_key_head_dim, tc.linear_num_key_heads, tc.rms_norm_eps);
    let v = v_raw.to_vec();

    Qwen35LinearQKV {q, k, v}
}

pub fn qwen35_compute_delta_gates(
    normed: &Vecf32,
    layer: &Qwen35LinearAttnLayer,
    tc: &TextConfig,
    gpu: &mut Gpu,
) -> Qwen35DeltaGates {

    let num_v_heads = tc.linear_num_value_heads;
    let alpha_raw = matmul_transposed_f16(normed, &layer.in_proj_a_w, num_v_heads, gpu);
    let beta_raw = matmul_transposed_f16(normed, &layer.in_proj_b_w, num_v_heads, gpu);

    let beta = beta_raw.iter().map(|&x| sigmoid(x)).collect();

    let decay = (0..num_v_heads).map(|i| {
        let dt = softplus(alpha_raw[i] + layer.dt_bias[i]);
        let a = -layer.a_log[i].exp();
        let d_res = (a * dt).exp();
        d_res
    }).collect();

    Qwen35DeltaGates {decay, beta}
}

fn state_vec_mul(
    state: &Arrf32,
    s_offset: usize,
    vec: &Arrf32,
    vec_offset: usize,
    k_dim: usize,
    v_dim: usize,
) -> Vecf32 {
    let mut out = vec![0.0f32; v_dim];
    for vi in 0..v_dim {
        for ki in 0..k_dim {
            out[vi] += state[s_offset + ki * v_dim + vi] * vec[vec_offset + ki];
        }
    }
    out
}

fn update_state(
    state: &mut Arrf32,
    s_offset: usize,
    k: &Arrf32,
    k_offset: usize,
    v: &Arrf32,
    v_offset: usize,
    s_k: &Arrf32,
    decay: f32,
    beta: f32,
    k_dim: usize,
    v_dim: usize,
) {
    for ki in 0..k_dim {
        let k_val = k[k_offset + ki];
        let bk = beta * k_val;

        for vi in 0..v_dim {
            let idx = s_offset + ki * v_dim + vi;
            let delta  = v[v_offset + vi] - s_k[vi];
            state[idx]= decay * state[idx] + bk * delta;
        }
    }
}

pub fn qwen35_recurrent_update(
    qkv: &Qwen35LinearQKV,
    gates: &Qwen35DeltaGates,
    recurrent_state: &mut Vecf32,
    tc: &TextConfig,
) -> Vecf32 {
    let num_v_heads = tc.linear_num_value_heads;
    let num_k_heads = tc.linear_num_key_heads;
    let k_dim = tc.linear_key_head_dim;
    let v_dim = tc.linear_value_head_dim;
    let heads_per_k = num_v_heads / num_k_heads;

    if recurrent_state.is_empty() {
        *recurrent_state = vec![0.0f32; num_v_heads * k_dim * v_dim];
    }

    let mut output = vec![0.0f32; num_v_heads * v_dim];

    for h in 0..num_v_heads {
        let k_h = h / heads_per_k;
        let k_offset = k_h * k_dim;
        let v_offset = h * v_dim;
        let s_offset = v_offset * k_dim;

        let s_k = state_vec_mul(recurrent_state, s_offset, &qkv.k, k_offset, k_dim, v_dim);
        update_state(recurrent_state, s_offset, &qkv.k, k_offset, &qkv.v, v_offset, &s_k, gates.decay[h], gates.beta[h], k_dim, v_dim);

        let head_out = state_vec_mul(recurrent_state, s_offset, &qkv.q, k_offset, k_dim, v_dim);

        output[v_offset..v_offset + v_dim].copy_from_slice(&head_out);
    }

    output
}

pub fn qwen35_linear_attn_output(
    recurrent_out: &Vecf32,
    normed: &Vecf32,
    layer: &Qwen35LinearAttnLayer,
    tc: &TextConfig,
    gpu: &mut Gpu,
) -> Vecf32 {
    let v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim;

    let z_raw = matmul_transposed_f16(normed, &layer.in_proj_z_w, v_dim, gpu);
    let z = silu(&z_raw);

    let head_dim = tc.linear_value_head_dim;
    let num_heads = tc.linear_num_value_heads;

    let mut gated = vec![0.0f32; v_dim];
    for h in 0..num_heads {
        let offset = h * head_dim;
        let head_slice = &recurrent_out[offset..offset + head_dim];
        let normed_head = rms_norm(head_slice, &layer.norm_w, tc.rms_norm_eps);

        for i in 0..head_dim {
            gated[offset + i] = normed_head[i] * z[offset + i];
        }
    }
    let res = matmul_transposed_f16(&gated, &layer.out_proj_w, tc.hidden_size, gpu);

    res
}

pub fn qwen35_linear_attention(
    hidden: &mut Vecf32,
    normed: &Vecf32,
    layer: &Qwen35LinearAttnLayer,
    tc: &TextConfig,
    kv_cache: &mut KVCache,
    layer_no: usize,
    gpu: &mut Gpu
) {
    let qkv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim;

    let qkv_proj = matmul_transposed_f16(normed, &layer.in_proj_qkv_w, qkv_dim, gpu);
    let qkv_conv = qwen35_conv1d_update(&qkv_proj, &mut kv_cache.conv_state[layer_no], &layer.conv1d_w, qkv_dim, tc.linear_conv_kernel_dim);
    let qkv_activated = silu(&qkv_conv);
    let qkv = qwen35_split_qkv(&qkv_activated, tc);
    let gates = qwen35_compute_delta_gates(normed, layer, tc, gpu);
    let recurrent_out = qwen35_recurrent_update(&qkv, &gates, &mut kv_cache.recurrent_state[layer_no], tc);
    let projected = qwen35_linear_attn_output(&recurrent_out, normed, layer, tc, gpu);

    add_to_vector(hidden, &projected);
}