mod safetensors;
mod norm;
mod embeddings;
mod matmul;
mod linear;
mod attention;
mod activation;
mod transformer;

use std::io::{Read, Seek};
use crate::embeddings::combine_embedding_weights;
use crate::matmul::{matmul, matmul_transposed};
use crate::norm::layer_norm;
use crate::safetensors::{load_layer, SafeTensor};
use crate::transformer::{transformer_pass, TransformerPassArgs};

fn main() {
    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");

    let token_id = 464;
    let dim = 768;

    let wte_start = token_id * dim;
    let wte_row = &wte[wte_start..wte_start + dim];
    let wpe_row = &wpe[0..dim];

    let mut hidden: Vec<f32> = combine_embedding_weights(wte_row, wpe_row);

    for i in 0..12 {
        let layer_weights = load_layer(&mut model, i);
        transformer_pass(TransformerPassArgs {hidden: &mut hidden, wte: &wte, wpe: &wpe, layer_weights: &layer_weights, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim});
    }

    hidden = layer_norm(&hidden, &ln_f_w, &ln_f_b);

    let vocab_size = 50257;
    let logits = matmul_transposed(&hidden, &wte, vocab_size);
    let mut top: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, v) in &top[..5] {
        println!("{} {:.4}", i, v)
    }

}
