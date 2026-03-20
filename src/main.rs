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
use crate::transformer::{transformer_multi, TransformerMultiArgs};

fn main() {
    let token_id = 464;
    let dim = 768;

    //temporary
    let num_layers = 1; //12

    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let ln_f_w = model.load_tensor("ln_f.weight");
    let ln_f_b = model.load_tensor("ln_f.bias");
    let layers = load_all_layers(&mut model, num_layers);


    let token_ids: Vec<usize> = vec![464, 3139, 286, 4881, 318];
    transformer_multi(TransformerMultiArgs {
        token_ids: &token_ids, num_layers, wte: &wte, wpe: &wpe, ln_f_w: & ln_f_w, ln_f_b: &ln_f_b, layers: &layers, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim
    })

    // test_single_token(TransformerArgs { token_id, num_layers, wte: &wte, wpe: &wpe, ln_f_w: & ln_f_w, ln_f_b: &ln_f_b, layers: &layers, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim });

}
