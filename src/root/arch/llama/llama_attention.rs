use crate::root::activation::softmax;
use crate::root::arch::llama::llama_config::LlamaConfig;
use crate::root::arch::llama::llama_layer::LlamaLayer;
use crate::root::attention::{compute_qk_scores, project_v_to_attn, ComputeQKScoresArgs};
use crate::root::common_types::Vecf32;
use crate::root::gpu::Gpu;
use crate::root::kv_cache::{KVCache, PositionKVVectors};
use crate::root::linear::add_to_vector;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::norm::rms_norm;
use crate::root::rope::{apply_rope, compute_rope_freqs, ApplyRopeArgs};

pub struct ComputeQKVArgs<'a> {
    normed: &'a mut Vecf32,
    layer: &'a LlamaLayer,
    config: &'a LlamaConfig,
    kv_cache: &'a mut KVCache,
    layer_no: usize,
    position: usize,
    gpu: &'a mut Gpu,
}

fn add_optional_bias(vec: &mut Vecf32, optional_bias: &Option<Vecf32>) {
    if let Some(bias) = optional_bias {
        add_to_vector(vec, &bias);
    }
}

pub struct ApplyQKNormArgs<'a> {
    vec: &'a mut Vecf32,
    norm_w: &'a Option<Vecf32>,
    num_heads: usize,
    head_dim: usize,
    eps: f32,
}
fn apply_qk_norm(args: ApplyQKNormArgs) {
    let ApplyQKNormArgs {vec, norm_w, num_heads, head_dim, eps} = args;

    if let Some(w) = norm_w {
        for h in 0..num_heads {
            let offset = h * head_dim;
            let head_slice = &vec[offset..offset+head_dim];
            let normed = rms_norm(head_slice, w, eps);
            vec[offset..offset+head_dim].copy_from_slice(&normed);
        }
    }
}

fn compute_qkv(args: ComputeQKVArgs<'_>) -> ( Vecf32,  &PositionKVVectors, &PositionKVVectors ) {
    let ComputeQKVArgs {normed, layer, config, kv_cache, layer_no, position, mut gpu} = args;
    let LlamaLayer {q_proj_w, q_proj_b, k_proj_w, k_proj_b, v_proj_w, v_proj_b, ..} = layer;
    let LlamaConfig {kv_dim, head_dim, rope_theta, attn_dim, num_attention_heads, rms_norm_eps, num_key_value_heads,  ..} = *config;

    let mut q = matmul_transposed_f16(&normed, q_proj_w, attn_dim, &mut gpu);
    add_optional_bias(&mut q, q_proj_b);

    let mut k = matmul_transposed_f16(&normed, k_proj_w, kv_dim, &mut gpu);
    add_optional_bias(&mut k, k_proj_b);

    apply_qk_norm(ApplyQKNormArgs { vec: &mut q, norm_w: &layer.q_norm_w, num_heads: num_attention_heads, head_dim, eps: rms_norm_eps});

    apply_qk_norm(ApplyQKNormArgs { vec: &mut k, norm_w: &layer.k_norm_w, num_heads: num_key_value_heads, head_dim, eps: rms_norm_eps});

    let mut v = matmul_transposed_f16(&normed, v_proj_w, kv_dim, &mut gpu);
    add_optional_bias(&mut v, v_proj_b);

    let (cos, sin) = compute_rope_freqs(head_dim, position, rope_theta);

    let mut q_roped = vec![0.0f32; q.len()];
    let q_num_heads = q.len() / head_dim;
    apply_rope(ApplyRopeArgs { out: &mut q_roped, x: &q, num_heads: q_num_heads, cos: &cos, sin: &sin, head_dim, nope_dim: 0, interleaved: false});

    let mut k_roped = vec![0.0f32; k.len()];
    let k_num_heads = k.len() / head_dim;
    apply_rope(ApplyRopeArgs { out: &mut k_roped, x: &k, num_heads: k_num_heads, cos: &cos, sin: &sin, head_dim, nope_dim: 0, interleaved: false});

    kv_cache.append(layer_no, k_roped, v);

    let k_cache = &kv_cache.k[layer_no];
    let v_cache = &kv_cache.v[layer_no];

    (q_roped, k_cache, v_cache)
}


pub struct LlamaGQAArgs<'a> {
    pub config: &'a LlamaConfig,
    pub kv_cache: &'a mut KVCache,
    pub gpu: &'a mut Gpu,
    pub layer: &'a LlamaLayer,
    pub hidden: &'a mut Vecf32,
    pub normed: &'a mut Vecf32,
    pub layer_no: usize,
    pub position: usize,
}

pub fn llama_gqa(args: LlamaGQAArgs) {
    let LlamaGQAArgs {config, gpu, layer, hidden, kv_cache, normed, layer_no, position } = args;

    let (q, k_cache, v_cache) = &compute_qkv(ComputeQKVArgs {
        normed, layer, config, kv_cache, layer_no, position, gpu
    });

    let cache_len = k_cache.len();

    let repeats = config.heads_to_kvheads;
    let mut attn_out = vec![0.0f32; config.attn_dim];

    for h in 0..config.num_attention_heads {
        let kv_h = h / repeats;
        let q_offset = h * config.head_dim;
        let kv_offset = kv_h * config.head_dim;

        let probs = compute_qk_scores(ComputeQKScoresArgs {kv_offset, head_dim: config.head_dim, k_cache, cache_len, q, q_offset});

        project_v_to_attn(&mut attn_out, &probs, &v_cache, cache_len, config.head_dim, kv_offset, q_offset);
    }

    let projected = matmul_transposed_f16(&attn_out, &layer.o_proj_w, config.hidden_size, gpu);

    add_to_vector(hidden, &projected);
}