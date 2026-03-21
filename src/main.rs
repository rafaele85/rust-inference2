mod safetensors;
mod norm;
mod embeddings;
mod matmul;
mod linear;
mod attention;
mod activation;
mod transformer;

use std::io::{Read, Seek};
use crate::activation::softmax;
use crate::attention::multi_head_attention;
use crate::linear::linear_projection;
use crate::safetensors::{load_all_layers, SafeTensor};
use crate::transformer::{fill_all_qkv, initialize_hidden, FillAllQKVArgs, InitializeHiddenArgs, TransformerMultiArgs};

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
    // test_multi_tokens(TransformerMultiArgs { token_ids: &token_ids, num_layers, wte: &wte, wpe: &wpe, ln_f_w: & ln_f_w, ln_f_b: &ln_f_b, layers: &layers, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim });

    // test_single_token(TransformerArgs { token_id, num_layers, wte: &wte, wpe: &wpe, ln_f_w: & ln_f_w, ln_f_b: &ln_f_b, layers: &layers, norm_final_weight: &ln_f_w, norm_final_bias: &ln_f_b, dim });

    let qkv_size = dim * 3;
    let seq_len = token_ids.len();
    let all_qkv_size = seq_len * qkv_size;
    let hidden_size = seq_len * dim;
    let n_heads = 12;
    let layer = &layers[0];
    let hidden = initialize_hidden(InitializeHiddenArgs {hidden_size, token_ids: &token_ids, wpe: &wpe, wte: &wte, dim});
    let all_qkv = fill_all_qkv(FillAllQKVArgs {seq_len, dim, hidden: &hidden, layer, all_qkv_size, qkv_size});
    let attn_raw = multi_head_attention(&*all_qkv, seq_len, dim, n_heads);

    let last_pos = seq_len - 1;
    let last_pos_offset = last_pos * dim;
    let last_attn = &attn_raw[last_pos_offset..last_pos_offset + dim];
    let attn_out = linear_projection(last_attn, &layer.proj_w, &layer.proj_b, dim);

    for v in &attn_out[0..10] {
        println!("{:.4}", v);
    }
}
