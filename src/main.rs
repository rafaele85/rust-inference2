mod safetensors;
mod norm;
mod embeddings;
mod matmul;
mod linear;
mod attention;
mod activation;

use std::io::{Read, Seek};
use crate::activation::gelu;
use crate::attention::extract_v;
use crate::embeddings::{combine_embedding_weights};
use crate::linear::{linear_projection, normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};
use crate::norm::layer_norm;
use crate::safetensors::{load_layer0, SafeTensor};

fn main() {
    let mut model = SafeTensor::open("model/model.safetensors");
    let wte = model.load_tensor("wte.weight");
    let wpe = model.load_tensor("wpe.weight");
    let (ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b) =
        load_layer0(&mut model);

    let token_id = 464;
    let dim = 768;
    let ffn_dim = 3072;
    let qkv_combined_dim = dim * 3; // all 3 weights are of the same size


    let wte_start = token_id * dim;
    let wte_row = &wte[wte_start..wte_start + dim];
    let wpe_row = &wpe[0..dim];

    let mut hidden: Vec<f32> = combine_embedding_weights(wte_row, wpe_row);

    let qkv = normalize_and_project(NormalizeAndProject {hidden: &mut hidden, norm_weights: &ln1_w, norm_bias: &ln1_b, projection_weights: &qkv_w, projection_bias: &qkv_b, projection_dim: qkv_combined_dim});

    let v = extract_v(&qkv);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &v, projection_weights: &proj_w, projection_bias: &proj_b, dim: dim, hidden: &mut hidden });

    let ffn_up = normalize_and_project(NormalizeAndProject {hidden: &mut hidden, norm_weights: &ln2_w, norm_bias: &ln2_b, projection_weights: &ffn_up_w, projection_bias: &ffn_up_b, projection_dim: ffn_dim});

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &ffn_act, projection_weights: &ffn_down_w, projection_bias: &ffn_down_b, dim: dim, hidden: &mut hidden });

    for i in 0..10 {
        let v = &hidden[i];
        println!("{:.4}", v)
    }
}
