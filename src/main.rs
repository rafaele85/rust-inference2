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
use crate::activation::softmax;
use crate::attention::multi_head_attention;
use crate::linear::linear_projection;
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{fill_all_qkv, initialize_hidden, transformer_multi, FillAllQKVArgs, InitializeHiddenArgs, TransformerMultiArgs};
use crate::vocab::Vocab;

fn main() {

    let num_layers = 12;

    let dim = 768;
    let n_heads = 12;
    let ffn_dim = dim * 4;
    let vocab_size = 50257;


    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");
    let layers = load_all_layers(&mut model, num_layers);
    let vocab = Vocab::load("model/vocab.json");


    let token_ids: Vec<usize> = vec![464, 3139, 286, 4881, 318];
    for &token_id in &token_ids {
        println!("input token_id: {} {}", token_id, vocab.decode(token_id));
    }

    let output_id = transformer_multi(TransformerMultiArgs {
        token_ids: &token_ids, layers: &layers, num_layers, ln_f_w: &ln_f_w, ln_f_b: &ln_f_b, wte: &wte, wpe: &wpe, dim, n_heads, vocab_size, ffn_dim
    });

    println!("output token_id: {} {}", output_id, vocab.decode(output_id));
}
