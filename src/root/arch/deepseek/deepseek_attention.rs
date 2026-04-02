use crate::root::arch::deepseek::deepseek_config::DeepseekConfig;
use crate::root::arch::deepseek::deepseek_layer::DeepseekLayer;
use crate::root::attention::{compute_qk_scores, project_v_to_attn, ComputeQKScoresArgs};
use crate::root::common_types::{Arrf32, Vecf32, WeightF16};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::{KVCache, PositionKVVectors};
use crate::root::linear::add_to_vector;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::norm::rms_norm;
use crate::root::rope::{apply_rope, apply_rope_to_head, compute_rope_freqs, ApplyRopeArgs, ApplyRopeToHeadArgs};

fn q_project(normed: &Vecf32, q_proj_w: &WeightF16, q_total: usize, gpu: &mut Gpu) -> Vecf32 {
    let res = matmul_transposed_f16(normed, q_proj_w, q_total, gpu);

    res
}

fn kv_down_project(normed: &Vecf32, kv_a_proj_w: &WeightF16, kv_a_total: usize, kv_lora_rank: usize, gpu: &mut Gpu) -> (Vecf32, Vecf32) {
    let kv_a = matmul_transposed_f16(normed, kv_a_proj_w, kv_a_total, gpu);
    let compressed_kv = kv_a[..kv_lora_rank].to_vec();
    let k_rope_shared = kv_a[kv_lora_rank..].to_vec();
    (compressed_kv, k_rope_shared)
}

fn kv_up_project(compressed_kv: &Vecf32, kv_a_layernorm_w: &Vecf32, kv_b_proj_w: &WeightF16, kv_b_total: usize, eps: f32, gpu: &mut Gpu) -> Vecf32 {
    let normed_kv = rms_norm(compressed_kv, kv_a_layernorm_w, eps);
    matmul_transposed_f16(&normed_kv, kv_b_proj_w, kv_b_total, gpu)
}

fn build_kv_for_caching(config: &DeepseekConfig, qk_head_dim: usize, kv_b: &Arrf32, k_rope_roped: &Arrf32) -> (Vecf32, Vecf32) {
    assert_eq!(k_rope_roped.len(), config.qk_rope_head_dim, "k_rope_roped length mismatch");
    let mut full_k = vec![0.0f32; config.num_attention_heads * qk_head_dim];
    let mut full_v = vec![0.0f32; config.num_attention_heads * config.v_head_dim];

    let h_dim = config.qk_nope_head_dim + config.v_head_dim;
    for h in 0..config.num_attention_heads {
        let kv_b_offset = h * h_dim;
        let k_offset = h * qk_head_dim;
        let k_rope_offset = k_offset + config.qk_nope_head_dim;
        let v_offset = h * config.v_head_dim;
        let v_src_offset = kv_b_offset + config.qk_nope_head_dim;

        let k_slice = &kv_b[kv_b_offset..kv_b_offset + config.qk_nope_head_dim];
        full_k[k_offset..k_offset + config.qk_nope_head_dim].copy_from_slice(k_slice);
        full_k[k_rope_offset..k_rope_offset + config.qk_rope_head_dim].copy_from_slice(&k_rope_roped);
        let v_slice = &kv_b[v_src_offset..v_src_offset + config.v_head_dim];
        full_v[v_offset..v_offset + config.v_head_dim].copy_from_slice(v_slice);
    }

    (full_k, full_v)
}

struct ApplyMLARopeResult  {
    q: Vecf32,
    k_rope_roped: Vecf32,
}

fn apply_mla_rope(q_raw: &Vecf32, k_rope_shared: &Vecf32, config: &DeepseekConfig, qk_head_dim: usize, position: usize) -> ApplyMLARopeResult {
    let (cos, sin) = compute_rope_freqs(config.qk_rope_head_dim, position, config.rope_theta);

    let mut q = q_raw.clone();

    let half = config.qk_rope_head_dim/2;
    apply_rope(ApplyRopeArgs {
        out: &mut q, x: &q_raw, num_heads: config.num_attention_heads, cos: &cos, sin: &sin, head_dim: qk_head_dim, nope_dim: config.qk_nope_head_dim, interleaved: true, half
    });

    let mut k_rope_roped = vec![0.0f32; config.qk_rope_head_dim];
    apply_rope_to_head(ApplyRopeToHeadArgs {x: &k_rope_shared, out: &mut k_rope_roped, half, offset: 0, cos: &cos, sin: &sin, interleaved: true});

    ApplyMLARopeResult {q, k_rope_roped}
}

fn mla_attention_scores(q: &Vecf32, k_cache: &PositionKVVectors, v_cache: &PositionKVVectors, config: &DeepseekConfig, qk_head_dim: usize) -> Vecf32 {
    let cache_len = k_cache.len();

    let mut attn_out = vec![0.0f32; config.num_attention_heads * config.v_head_dim];

    for h in 0..config.num_attention_heads {
        let q_offset = h * qk_head_dim;
        let probs = compute_qk_scores(ComputeQKScoresArgs {cache_len, kv_offset: q_offset, q_offset, head_dim: qk_head_dim, q: &q, k_cache});

        let v_offset = h * config.v_head_dim;
        project_v_to_attn(&mut attn_out, &probs, &v_cache, cache_len, config.v_head_dim, v_offset, v_offset);
    }

    attn_out
}

pub struct DeepseekMLAAttentionArgs<'a> {
    pub hidden: &'a mut Vecf32,
    pub normed: &'a mut Vecf32,
    pub layer: &'a DeepseekLayer,
    pub config: &'a DeepseekConfig,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
    pub position: usize,
    pub gpu: &'a mut Gpu,
}
pub fn deepseek_mla_attention(args: DeepseekMLAAttentionArgs) {
    let DeepseekMLAAttentionArgs {hidden, normed, layer, config, kv_cache, layer_no, position, gpu} = args;
    let DeepseekLayer {q_proj_w, kv_a_proj_w, kv_a_layernorm_w, kv_b_proj_w, ..} = layer;
    let qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    let q_total = config.num_attention_heads * qk_head_dim;
    let kv_a_total = config.kv_lora_rank + config.qk_rope_head_dim;
    let kv_b_total = config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim);

    let q_raw = q_project(normed, q_proj_w, q_total, gpu);
    let (compressed_kv, k_rope_shared) = kv_down_project(normed, kv_a_proj_w, kv_a_total, config.kv_lora_rank, gpu);

    let kv_b  = kv_up_project(&compressed_kv, &kv_a_layernorm_w, &kv_b_proj_w, kv_b_total, config.rms_norm_eps, gpu);

    let ApplyMLARopeResult {q, k_rope_roped} = apply_mla_rope(&q_raw, &k_rope_shared, config, qk_head_dim, position);

    let (full_k, full_v) = build_kv_for_caching(config, qk_head_dim, &kv_b, &k_rope_roped);

    kv_cache.append(layer_no, full_k, full_v);

    let attn_out = mla_attention_scores(&q, &kv_cache.k[layer_no], &kv_cache.v[layer_no], config, qk_head_dim);

    let projected = matmul_transposed_f16(&attn_out, &layer.o_proj_w, config.hidden_size, gpu);

    add_to_vector(hidden, &projected);
}

