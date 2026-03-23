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
mod rope;
mod model;

use crate::activation::silu;
use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::matmul::matmul_transposed;
use crate::norm::rms_norm;
use crate::rope::{apply_rope, compute_rope_freqs};
use model::safetensors::SafeTensor;
use crate::common_types::Vecf32;
use crate::linear::add_to_vector;
use crate::model::gpt2::gpt2::GPT2Model;
use crate::model::llama::llama::LlamaModel;
use crate::model::llama::llama_transformer::{llama_layer, LlamaLayerArgs};
use crate::transformer::{generate_token, GenerateTokenArgs};

fn main() {
    let gpu = Gpu::new(0);

    let mut safe_tensor_llama = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6");

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, &gpu);
    let token_id = 1usize;
    let token_start = token_id * llama_model.config.hidden_size;
    let embed_token = &llama_model.embed_weights[token_start..token_start + llama_model.config.hidden_size];


    let mut hidden_llama = embed_token.to_vec();

    let mut kv_cache_llama = KVCache::new(llama_model.config.num_hidden_layers);
    llama_layer(LlamaLayerArgs {
        kv_cache: &mut kv_cache_llama,
        layer: &llama_model.layers[0],
        config: &llama_model.config,
        layer_no: 0,
        position: 0,
        gpu: &gpu,
        hidden: &mut hidden_llama
    });

    println!("layer0_out[:5]: {:?}", &hidden_llama[0..5]);


    //-----------GPT2-------------

    let mut safe_tensor_gpt2 = SafeTensor::open("model");
    let gpt2_model = GPT2Model::new(&mut safe_tensor_gpt2, &gpu);

    println!("Begin generation");

    let mut kv_cache = KVCache::new(gpt2_model.config.n_layer);

    let mut token_ids: Vec<usize> = gpt2_model.vocab.encode("The capital of France is");
    let seq_len = token_ids.len();

    let mut first_generated = 0;
    for pos in 0..seq_len {
        let id = token_ids[pos];
        let top_id = generate_token(GenerateTokenArgs {
            token_id: id, position: pos,
            kv_cache: &mut kv_cache, gpu: &gpu, gpt2_model: &gpt2_model
        });
        first_generated = top_id;
    }
    token_ids.push(first_generated);

    for _ in 0..49 {
        let seq_len = token_ids.len();
        let pos = seq_len - 1;
        let last_id = *token_ids.last().unwrap();
        let top_id = generate_token(GenerateTokenArgs {
            token_id: last_id, position: pos,
            gpt2_model: &gpt2_model, kv_cache: &mut kv_cache, gpu: &gpu
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| gpt2_model.vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
