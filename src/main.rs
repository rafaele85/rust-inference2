mod safetensors;
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

use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::norm::rms_norm;
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{generate_token, GenerateTokenArgs};
use crate::vocab::Vocab;

fn main() {
    let mut llama_model = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/model.safetensors");
    let embed_weights = llama_model.load_tensor_bf16("model.embed_tokens.weight");
    let ln1_w = llama_model.load_tensor_bf16("model.layers.0.input_layernorm.weight");
    let dim = 2048;
    let token_id = 1usize;
    let embed_token = &embed_weights[token_id * dim..token_id * dim + dim];
    println!("embed_token first 5: {:?}", &embed_token[0..5]);

    let normed = rms_norm(embed_token, &ln1_w, 1e-5);
    println!("rms_norm {:?}", &normed[0..5]);

    let gpu = Gpu::new(0);

    let num_layers = 12;

    let dim = 768;
    let n_heads = 12;
    let ffn_dim = dim * 4;

    let vocab = Vocab::load("model/vocab.json");

    let vocab_size = vocab.size();


    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wte_gpu = gpu.upload(&wte);
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");
    let layers = load_all_layers(&mut model, num_layers, &gpu);

    println!("Begin generation");

    let mut kv_cache = KVCache::new(num_layers);

    let mut token_ids: Vec<usize> = vocab.encode("The capital of France is");
    let seq_len = token_ids.len();
    let head_dim = dim / n_heads;

    let mut first_generated = 0;
    for pos in 0..seq_len {
        let id = token_ids[pos];
        let top_id = generate_token(GenerateTokenArgs {
            token_id: id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers, wte: &wte, wte_gpu: &wte_gpu, wpe: &wpe,
            ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, layers: &layers, kv_cache: &mut kv_cache, head_dim, gpu: &gpu
        });
        first_generated = top_id;
    }
    token_ids.push(first_generated);

    for _ in 0..49 {
        let seq_len = token_ids.len();
        let pos = seq_len - 1;
        let last_id = *token_ids.last().unwrap();
        let top_id = generate_token(GenerateTokenArgs {
            token_id: last_id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers, wte: &wte, wte_gpu: &wte_gpu, wpe: &wpe,
            ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, layers: &layers, kv_cache: &mut kv_cache, head_dim, gpu: &gpu
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
