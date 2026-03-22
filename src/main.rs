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

fn main() {
    let gpu = Gpu::new(0);

    let mut safe_tensor_llama = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6");

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, &gpu);

    let input_norm_w = &llama_model.layers[0].input_norm_w;
    let q_proj_gpu = &llama_model.layers[0].q_proj_w;
    let k_proj_gpu = &llama_model.layers[0].k_proj_w;
    let v_proj_gpu = &llama_model.layers[0].v_proj_w;
    let gate_proj_gpu = &llama_model.layers[0].gate_proj_w;


    let token_id = 1usize;
    let embed_token = &llama_model.embed_weights[token_id * llama_model.config.hidden_size..token_id * llama_model.config.hidden_size + llama_model.config.hidden_size];
    println!("embed_token first 5: {:?}", &embed_token[0..5]);

    let normed = rms_norm(embed_token, &input_norm_w, 1e-5);
    println!("rms_norm {:?}", &normed[0..5]);

    let q = matmul_transposed(&normed, &q_proj_gpu, llama_model.config.hidden_size, &gpu);
    println!("q {:?}", &q[0..5]);

    let gate = matmul_transposed(&normed, &gate_proj_gpu, llama_model.config.intermediate_size, &gpu);
    let silu_gate = silu(&gate);
    println!("silu_gate {:?}", &silu_gate[0..5]);

    let (cos, sin) = compute_rope_freqs(llama_model.config.head_dim, 1, llama_model.config.rope_theta, llama_model.config.head_dim/2);
    println!("cos {:?}", &cos[0..5]);
    println!("sin {:?}", &sin[0..5]);

    let q_rope = apply_rope(&q, &cos, &sin, llama_model.config.head_dim);
    println!("q_rope {:?}", &q_rope[0..5]);

    let k = matmul_transposed(&normed, &k_proj_gpu, llama_model.config.kv_dim, &gpu);
    let k_rope = apply_rope(&k, &cos, &sin, llama_model.config.head_dim );
    println!("k_rope {:?}", &k_rope[0..5]);

    let v = matmul_transposed(&normed, v_proj_gpu, llama_model.config.kv_dim, &gpu);
    println!("v[:5]: {:?}", &v[0..5]);

    //-----------GPT2-------------

    let mut safe_tensor_gpt2 = SafeTensor::open("model");
    let gpt2_model = GPT2Model::new(&mut safe_tensor_gpt2, &gpu);

    println!("Begin generation");

    let mut kv_cache = KVCache::new(gpt2_model.config.n_layer);

    let mut token_ids: Vec<usize> = gpt2_model.vocab.encode("The capital of France is");
    let seq_len = token_ids.len();

    let mut first_generated = 0;
    for pos in 0..seq_len {
        let id = token_ids[pos];
        let top_id = generate_token(GenerateTokenArgs {
            token_id: id, position: pos,
            kv_cache: &mut kv_cache, gpu: &gpu, gpt2_model: &gpt2_model
        });
        first_generated = top_id;
    }
    token_ids.push(first_generated);

    for _ in 0..49 {
        let seq_len = token_ids.len();
        let pos = seq_len - 1;
        let last_id = *token_ids.last().unwrap();
        let top_id = generate_token(GenerateTokenArgs {
            token_id: last_id, position: pos,
            gpt2_model: &gpt2_model, kv_cache: &mut kv_cache, gpu: &gpu
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| gpt2_model.vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
