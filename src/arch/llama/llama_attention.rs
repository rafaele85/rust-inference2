use crate::activation::softmax;
use crate::common_types::Vecf32;
use crate::gpu::Gpu;
use crate::kv_cache::{KVCache, PositionKVVectors};
use crate::linear::add_to_vector;
use crate::matmul::matmul_transposed;
use crate::arch::llama::llama_config::LlamaConfig;
use crate::arch::llama::llama_layer::LlamaLayer;
use crate::arch::llama::rope::{apply_rope, compute_rope_freqs};

pub struct ComputeQKVArgs<'a> {
    normed: &'a mut Vecf32,
    layer: &'a LlamaLayer,
    config: &'a LlamaConfig,
    kv_cache: &'a mut KVCache,
    layer_no: usize,
    position: usize,
    gpu: &'a Gpu,
}



fn compute_qkv(args: ComputeQKVArgs<'_>) -> ( Vecf32,  &PositionKVVectors, &PositionKVVectors ) {
    let ComputeQKVArgs {normed, layer, config, kv_cache, layer_no, position, gpu} = args;
    let q = matmul_transposed(&normed, &layer.q_proj_w, config.hidden_size, gpu);
    let k = matmul_transposed(&normed, &layer.k_proj_w, config.kv_dim, gpu);
    let v = matmul_transposed(&normed, &layer.v_proj_w, config.kv_dim, gpu);

    let (cos, sin) = compute_rope_freqs(config.head_dim, position, config.rope_theta, config.head_dim/2);
    let q_roped = apply_rope(&q, &cos, &sin, config.head_dim);
    let k_roped = apply_rope(&k, &cos, &sin, config.head_dim);
    kv_cache.append(layer_no, k_roped, v);

    let k_cache = &kv_cache.k[layer_no];
    let v_cache = &kv_cache.v[layer_no];

    (q_roped, k_cache, v_cache)
}


pub struct LlamaGQAArgs<'a> {
    pub config: &'a LlamaConfig,
    pub kv_cache: &'a mut KVCache,
    pub gpu: &'a Gpu,
    pub layer: &'a LlamaLayer,
    pub hidden: &'a mut Vecf32,
    pub normed: &'a mut Vecf32,
    pub layer_no: usize,
    pub position: usize,
}

pub fn llama_gqa(args: LlamaGQAArgs) {
    let LlamaGQAArgs {config, gpu, layer, hidden, kv_cache, normed, layer_no, position } = args;

    let (q, k_cache, v_cache) = compute_qkv(ComputeQKVArgs {
        normed, layer, config, kv_cache, layer_no, position, gpu
    });

    let cache_len = k_cache.len();

    let repeats = config.heads_to_kvheads;
    let mut attn_out = vec![0.0f32; config.hidden_size];

    for h in 0..config.num_attention_heads {
        let kv_h = h / repeats;
        let q_offset = h * config.head_dim;
        let kv_offset = kv_h * config.head_dim;

        let mut scores = vec![0.0f32; cache_len];
        for pos in 0..cache_len {
            let mut dot = 0.0f32;
            for i in 0..config.head_dim {
                dot += q[q_offset + i] * k_cache[pos][kv_offset + i];
            }
            scores[pos] = dot / (config.head_dim as f32).sqrt();
        }

        let probs = softmax(&scores);

        for pos in 0..cache_len {
            for i in 0..config.head_dim {
                attn_out[q_offset+i] += probs[pos] * v_cache[pos][kv_offset+i];
            }
        }
    }

    let projected = matmul_transposed(&attn_out, &layer.o_proj_w, config.hidden_size, gpu);

    add_to_vector(hidden, &projected);
}