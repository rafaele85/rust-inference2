mod safetensors;
mod norm;
mod embeddings;
mod matmul;
mod linear;
mod attention;
mod activation;
mod transformer;

use std::io::{Read, Seek};
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{transformer, TransformerArgs};

fn main() {
    let num_layers = 12;
    let token_id = 464;
    let dim = 768;

    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");
    let layers = load_all_layers(&mut model, num_layers);


    let top_id = transformer(TransformerArgs {
        token_id, num_layers, wte: &wte, wpe: &wpe, ln_f_w: & ln_f_w, ln_f_b: &ln_f_b, layers: &layers, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim
    });

    println!("{}", top_id);
}
