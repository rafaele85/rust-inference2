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

use std::io::{Read, Seek};
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{transformer_multi, TransformerMultiArgs};
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

    let mut token_ids: Vec<usize> = vocab.encode("The capital of France is");

    for _ in 0..5 {
        let output_id = transformer_multi(TransformerMultiArgs {
            token_ids: &token_ids, layers: &layers, num_layers, ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, wte: &wte, wpe: &wpe, dim, n_heads, vocab_size, ffn_dim
        });
        token_ids.push(output_id);
    }

    let output = token_ids.iter().map(|id| vocab.decode(*id)).collect::<Vec<_>>();
    let text = output.join("");
    println!("{}", text);
}
