mod norm;
mod embeddings;
mod matmul;
mod linear;
mod attention;
mod activation;
mod transformer;
mod logits;
mod vocab;
mod kv_cache;
mod common_types;
mod gpu;
mod rope;
mod model;

use crate::activation::silu;
use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::matmul::matmul_transposed;
use crate::norm::rms_norm;
use crate::rope::{apply_rope, compute_rope_freqs};
use model::safetensors::SafeTensor;
use crate::model::gpt2::gpt2::GPT2Model;
use crate::model::llama::llama::LlamaModel;
use crate::transformer::{generate_token, GenerateTokenArgs};
use crate::vocab::Vocab;

fn main() {
    let gpu = Gpu::new(0);

    let mut safe_tensor_llama = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors");
    let num_layers = 1;
    let llama_model = LlamaModel::new(&mut safe_tensor_llama, num_layers, &gpu);
    let embed_weights = safe_tensor_llama.load_tensor_bf16("model.embed_tokens.weight");

    let input_norm_w = &llama_model.layers[0].input_norm_w;
    let q_proj_gpu = &llama_model.layers[0].q_proj_w;
    let k_proj_gpu = &llama_model.layers[0].k_proj_w;
    let gate_proj_gpu = &llama_model.layers[0].gate_proj_w;

    //let q_proj_w = safe_tensor_llama.load_tensor_bf16("model.layers.0.self_attn.q_proj.weight");
    //let gate_proj_w = safe_tensor_llama.load_tensor_bf16("model.layers.0.mlp.gate_proj.weight");
    //let k_proj_w = safe_tensor_llama.load_tensor_bf16("model.layers.0.self_attn.k_proj.weight");


    let dim = 2048;
    let token_id = 1usize;
    let embed_token = &embed_weights[token_id * dim..token_id * dim + dim];
    println!("embed_token first 5: {:?}", &embed_token[0..5]);

    let normed = rms_norm(embed_token, &input_norm_w, 1e-5);
    println!("rms_norm {:?}", &normed[0..5]);

    let q = matmul_transposed(&normed, &q_proj_gpu, dim, &gpu);
    println!("q {:?}", &q[0..5]);

    let gate = matmul_transposed(&normed, &gate_proj_gpu, 5632, &gpu);
    let silu_gate = silu(&gate);
    println!("silu_gate {:?}", &silu_gate[0..5]);

    let (cos, sin) = compute_rope_freqs(64, 1, 10000.0, 64/2);
    println!("cos {:?}", &cos[0..5]);
    println!("sin {:?}", &sin[0..5]);

    let q_rope = apply_rope(&q, &cos, &sin, 64);
    println!("q_rope {:?}", &q_rope[0..5]);

    let k = matmul_transposed(&normed, &k_proj_gpu, 256, &gpu);
    let k_rope = apply_rope(&k, &cos, &sin, 64);
    println!("k_rope {:?}", &k_rope[0..5]);




    let num_layers = 12;

    let dim = 768;
    let n_heads = 12;
    let ffn_dim = dim * 4;

    let vocab = Vocab::load("model/vocab.json");

    let vocab_size = vocab.size();


    let mut safe_tensor_gpt2 = SafeTensor::open("model/model.safetensors");
    let gpt2_model = GPT2Model::new(&mut safe_tensor_gpt2, &gpu, num_layers);

    println!("Begin generation");

    let mut kv_cache = KVCache::new(num_layers);

    let mut token_ids: Vec<usize> = vocab.encode("The capital of France is");
    let seq_len = token_ids.len();
    let head_dim = dim / n_heads;

    let mut first_generated = 0;
    for pos in 0..seq_len {
        let id = token_ids[pos];
        let top_id = generate_token(GenerateTokenArgs {
            token_id: id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers,
            kv_cache: &mut kv_cache, head_dim, gpu: &gpu, gpt2_model: &gpt2_model
        });
        first_generated = top_id;
    }
    token_ids.push(first_generated);

    for _ in 0..49 {
        let seq_len = token_ids.len();
        let pos = seq_len - 1;
        let last_id = *token_ids.last().unwrap();
        let top_id = generate_token(GenerateTokenArgs {
            token_id: last_id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers,
            gpt2_model: &gpt2_model, kv_cache: &mut kv_cache, head_dim, gpu: &gpu
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
