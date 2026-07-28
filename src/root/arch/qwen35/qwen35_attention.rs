use crate::root::activation::sigmoid;
use crate::root::arch::qwen35::qwen35_config::TextConfig;
use crate::root::arch::qwen35::qwen35_layer::Qwen35FullAttnLayer;
use crate::root::attention::{gqa_attention};
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::{KVCache, PositionKVVectors};
use crate::root::linear::add_to_vector;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::norm::{rms_norm, rms_norm_zero_centered};
use crate::root::rope::{apply_rope, compute_rope_freqs, ApplyRopeArgs};

fn qwen35_compute_q_k(
    raw: &Arrf32,
    norm_w: &Arrf32,
    head_dim: usize,
    num_heads: usize,
    rms_norm_eps: f32,
) -> Vecf32 {
    let dim = num_heads * head_dim;

    let mut out = vec![0.0f32; dim];
    for h in 0..num_heads {
        let offset = h * head_dim;
        let head_slice = &raw[offset..offset + head_dim];
        let normed_head = rms_norm_zero_centered(head_slice, norm_w, rms_norm_eps);
        out[offset..offset+head_dim].copy_from_slice(&normed_head);
    }

    out
}


pub struct Qwen35KV {
    pub q: Vecf32,
    pub k: Vecf32,
    pub v: Vecf32,
    pub gate: Vecf32,
}
pub fn qwen35_compute_qkv(
    normed: &Vecf32,
    layer: &Qwen35FullAttnLayer,
    tc: &TextConfig,
    gpu: &mut Gpu,
) -> Qwen35KV {
    let num_heads = tc.num_attention_heads;
    let num_kv_heads = tc.num_key_value_heads;
    let head_dim = tc.head_dim;
    let attn_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Q proj outputs 2x: [num_heads * head_dim * 2]
    // Layout per head: [q(head_dim), gate(head_dim)]
    let q_gate_raw = matmul_transposed_f16(normed, &layer.q_proj_w, attn_dim * 2, gpu);

    let mut q_raw = vec![0.0f32; attn_dim];
    let mut gate = vec![0.0f32; attn_dim];
    for h in 0..num_heads {
        let src = h * head_dim * 2;
        let dst = h * head_dim;

        q_raw[dst..dst + head_dim].copy_from_slice(&q_gate_raw[src..src + head_dim]);

        gate[dst..dst + head_dim].copy_from_slice(&q_gate_raw[src + head_dim..src + head_dim * 2]);
    }


    // K and V are separate projections, no deinterleave needed
    let k_raw = matmul_transposed_f16(normed, &layer.k_proj_w, kv_dim, gpu);
    let v = matmul_transposed_f16(normed, &layer.v_proj_w, kv_dim, gpu);

    // Per-head QK norms (zero-centered for AWQ weights)
    let q = qwen35_compute_q_k(&q_raw, &layer.q_norm_w, head_dim, num_heads, tc.rms_norm_eps);
    let k = qwen35_compute_q_k(&k_raw, &layer.k_norm_w, head_dim, num_kv_heads, tc.rms_norm_eps);

    Qwen35KV { q, k, v, gate }
}
pub fn qwen35_apply_rope(
    q: &mut Vecf32,
    k: &mut Vecf32,
    tc: &TextConfig,
    position: usize,
) {
    let rope_dim = (tc.head_dim as f32 * tc.rope_parameters.partial_rotary_factor) as usize;
    let nope_dim = 0;
    let half = rope_dim / 2;
    let (cos, sin) = compute_rope_freqs(half, position, tc.rope_parameters.rope_theta);

    let q_clone = q.clone();
    apply_rope(ApplyRopeArgs {
        out: q, x: &q_clone, num_heads: tc.num_attention_heads, cos: &cos, sin: &sin, head_dim: tc.head_dim, nope_dim, interleaved: false,
    });

    let k_clone = k.clone();
    apply_rope(ApplyRopeArgs {
        out: k, x: &k_clone, num_heads: tc.num_key_value_heads, cos: &cos, sin: &sin, head_dim: tc.head_dim, nope_dim, interleaved: false,
    });
}


pub fn qwen35_gqa_attention(
    q: &Vecf32,
    k_cache: &PositionKVVectors,
    v_cache: &PositionKVVectors,
    tc: &TextConfig,
) -> Vecf32 {

    let attn_dim = tc.num_attention_heads * tc.head_dim;
    let repeats = tc.num_attention_heads / tc.num_key_value_heads;

    let res = gqa_attention(q, k_cache, v_cache, tc, attn_dim, repeats);

    res
}

pub fn qwen35_gate_and_project(
    attn_out: &Vecf32,
    gate: &Vecf32,
    o_proj_w: &WeightF16,
    hidden_size: usize,
    gpu: &mut Gpu,
) -> Vecf32 {
    let attn_dim = attn_out.len();
    let mut gated = vec![0.0; attn_dim];

    for i in 0..attn_dim {
        let sig = sigmoid(gate[i]);
        gated[i] = attn_out[i] * sig;
    }
    let res = matmul_transposed_f16(&gated, o_proj_w, hidden_size, gpu);

    res
}

pub fn qwen35_full_attention(
    hidden: &mut Vecf32,
    normed: &Vecf32,
    layer: &Qwen35FullAttnLayer,
    tc: &TextConfig,
    kv_cache: &mut KVCache,
    layer_no: usize,
    position: usize,
    gpu: &mut Gpu,
) {
    let mut qkv = qwen35_compute_qkv(normed, layer, tc, gpu);
    qwen35_apply_rope(&mut qkv.q, &mut qkv.k, tc, position);

    kv_cache.append(layer_no, qkv.k, qkv.v);

    let attn_out = qwen35_gqa_attention(&qkv.q, &kv_cache.k[layer_no], &kv_cache.v[layer_no], tc);
    let projected = qwen35_gate_and_project(&attn_out, &qkv.gate, &layer.o_proj_w, tc.hidden_size, gpu);

    add_to_vector(hidden, &projected);
}

