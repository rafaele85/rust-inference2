use crate::activation::gelu;
use crate::attention::{extract_q, extract_v};
use crate::embeddings::combine_embedding_weights;
use crate::linear::{linear_projection, normalize_and_project, project_and_residual, NormalizeAndProject, ProjectAndResidualArgs};
use crate::matmul::matmul_transposed;
use crate::norm::layer_norm;
use crate::safetensors::Layer;

pub struct TransformerPassArgs<'a> {
    pub hidden: &'a mut Vec<f32>,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layer: &'a Layer,
    pub norm_final_weight: &'a Vec<f32>,
    pub norm_final_bias: &'a Vec<f32>,
    pub dim: usize,
}

pub fn transformer_pass(args: TransformerPassArgs) {
    let ffn_dim = 3072;
    let qkv_combined_dim = args.dim * 3; // all 3 weights are of the same size

    let layer = &args.layer;

    let qkv = normalize_and_project(NormalizeAndProject {
        hidden: args.hidden, norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b, projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_combined_dim
    });

    let v = extract_v(&qkv);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &v, projection_weights: &layer.proj_w, projection_bias: &layer.proj_b, dim: args.dim, hidden: args.hidden });

    let ffn_up = normalize_and_project(NormalizeAndProject {
        hidden: args.hidden, norm_weights: &layer.ln2_w, norm_bias: &layer.ln2_b, projection_weights: &layer.ffn_up_w, projection_bias: &layer.ffn_up_b, projection_dim: ffn_dim
    });

    let ffn_act = gelu(&ffn_up);

    project_and_residual(ProjectAndResidualArgs {activation_weights: &ffn_act, projection_weights: &layer.ffn_down_w, projection_bias: &layer.ffn_down_b, dim: args.dim, hidden: args.hidden });
}

pub struct TransformerArgs<'a> {
    pub num_layers: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layers: &'a Vec<Layer>,
    pub norm_final_weight: &'a Vec<f32>,
    pub norm_final_bias: &'a Vec<f32>,
    pub dim: usize,
    pub token_id: usize,
    pub ln_f_w: &'a Vec<f32>,
    pub ln_f_b: &'a Vec<f32>,
}

pub fn transformer(args: TransformerArgs) -> usize {
    let wte_start = args.token_id * args.dim;
    let wte_row = &args.wte[wte_start..wte_start + args.dim];
    let wpe_row = &args.wpe[0..args.dim];

    let mut hidden: Vec<f32> = combine_embedding_weights(wte_row, wpe_row);

    for i in 0..args.num_layers {
        transformer_pass(TransformerPassArgs {
            hidden: &mut hidden, wte: args.wte, wpe: args.wpe, layer: &args.layers[i], norm_final_weight: args.norm_final_weight, norm_final_bias: args.norm_final_bias, dim: args.dim
        });
    }

    hidden = layer_norm(&hidden, &args.ln_f_w, &args.ln_f_b);

    let vocab_size = 50257;
    let logits = matmul_transposed(&hidden, &args.wte, vocab_size);
    let top_id = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;

    top_id
}

pub struct InitializeHiddenArgs<'a> {
    pub hidden_size: usize,
    pub dim: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub token_ids: &'a Vec<usize>
}
pub fn initialize_hidden(args: InitializeHiddenArgs) -> Vec<f32> {
    let InitializeHiddenArgs {hidden_size, dim, wte, wpe, token_ids} = args;
    let mut hidden = vec![0.0f32; hidden_size];

    for (pos, &token_id) in token_ids.iter().enumerate() {
        for i in 0..dim {
            let offset = pos * dim;
            hidden[offset + i] = wte[token_id * dim + i] + wpe[offset + i];
        }
    }

    hidden
}


pub struct TransformerMultiArgs<'a> {
    pub num_layers: usize,
    pub wte: &'a Vec<f32>,
    pub wpe: &'a Vec<f32>,
    pub layers: &'a Vec<Layer>,
    pub norm_final_weight: &'a Vec<f32>,
    pub norm_final_bias: &'a Vec<f32>,
    pub dim: usize,
    pub token_ids: &'a Vec<usize>,
    pub ln_f_w: &'a Vec<f32>,
    pub ln_f_b: &'a Vec<f32>,
}
pub fn transformer_multi(args: TransformerMultiArgs) {
    let TransformerMultiArgs {
        dim, norm_final_weight,norm_final_bias, num_layers, layers, ln_f_w, ln_f_b, wte, wpe,  token_ids
    } = args;
    let seq_len = token_ids.len();
    let last_pos = seq_len - 1;
    let qkv_size = dim * 3;
    let all_qkv_size = seq_len * qkv_size;
    let last_pos_offset = last_pos * qkv_size;
    let hidden_size = seq_len * dim;

    let hidden = initialize_hidden(InitializeHiddenArgs {hidden_size, dim, wpe: &wpe, wte: &wte, token_ids});

    for layer_no in 0..num_layers {
        let layer = &layers[layer_no];
        let all_qkv = fill_all_qkv(FillAllQKVArgs {all_qkv_size, qkv_size, hidden: &hidden, seq_len, dim, layer});

        let last_qkv = &all_qkv[last_pos_offset..last_pos_offset + qkv_size];
        let q = extract_q(last_qkv);
        for v in &q[..10] {
            println!("{:.4}", v);
        }
    }
}

pub struct FillAllQKVArgs<'a> {
    pub all_qkv_size: usize,
    pub seq_len: usize,
    pub dim: usize,
    pub hidden: &'a Vec<f32>,
    pub layer: &'a Layer,
    pub qkv_size: usize,
}
pub fn fill_all_qkv(args: FillAllQKVArgs) -> Vec<f32> {
    let FillAllQKVArgs {all_qkv_size, seq_len, dim, hidden, layer, qkv_size} = args;
    let mut all_qkv = vec![0.0f32; all_qkv_size];
    for pos in 0..seq_len {
        let offset = pos * dim;
        let mut h = &hidden[offset..offset + dim];
        let qkv = normalize_and_project(NormalizeAndProject {
            hidden: &mut h.to_vec(), norm_weights: &layer.ln1_w, norm_bias: &layer.ln1_b, projection_weights: &layer.qkv_w, projection_bias: &layer.qkv_b, projection_dim: qkv_size
        });
        let qkv_offset = pos * qkv_size;
        all_qkv[qkv_offset..qkv_offset + qkv_size].copy_from_slice(&qkv);
    }

    all_qkv
}