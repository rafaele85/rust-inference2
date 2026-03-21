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

use std::io::{Read, Seek};
use crate::kv_cache::KVCache;
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{generate_token, GenerateTokenArgs};
use crate::vocab::Vocab;

fn main() {

    let num_layers = 12;

    let dim = 768;
    let n_heads = 12;
    let ffn_dim = dim * 4;

    let vocab = Vocab::load("model/vocab.json");

    let vocab_size = vocab.size();


    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");
    let layers = load_all_layers(&mut model, num_layers);

    println!("Begin generation");

    let mut kv_cache = KVCache::new(num_layers);

    let mut token_ids: Vec<usize> = vocab.encode("The capital of France is");
    let seq_len = token_ids.len();
    let head_dim = dim / n_heads;

    let mut first_generated = 0;
    for pos in 0..seq_len {
        let id = token_ids[pos];
        let top_id = generate_token(GenerateTokenArgs {
            token_id: id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers, wte: &wte, wpe: &wpe,
            ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, layers: &layers, kv_cache: &mut kv_cache, head_dim, seq_len
        });
        first_generated = top_id;
    }
    token_ids.push(first_generated);

    for _ in 0..4 {
        let seq_len = token_ids.len();
        let pos = seq_len - 1;
        let last_id = *token_ids.last().unwrap();
        let top_id = generate_token(GenerateTokenArgs {
            token_id: last_id, position: pos, dim, n_heads, ffn_dim, vocab_size, num_layers, wte: &wte, wpe: &wpe,
            ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, layers: &layers, kv_cache: &mut kv_cache, head_dim, seq_len
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
