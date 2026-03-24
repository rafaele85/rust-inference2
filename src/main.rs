mod matmul;
mod activation;
mod kv_cache;
mod common_types;
mod gpu;
mod model;
mod linear;
mod tokenizer;

use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use model::safetensors::SafeTensor;
use crate::model::gpt2::gpt2::GPT2Model;
use crate::model::llama::llama::LlamaModel;
use crate::model::llama::llama_transformer::{llama_generate_token, LlamaGenerateTokenArgs};
use model::gpt2::gpt2_transformer::{gpt2_generate_token, GPT2GenerateTokenArgs};
use crate::tokenizer::BpeTokenizer;

fn main() {
    let gpu = Gpu::new(0);

    let gpt2_tokenizer = BpeTokenizer::load("model/tokenizer.json");
    println!("GPT-2 vocab size: {}", gpt2_tokenizer.vocab.len());

    let llama_tokenizer = BpeTokenizer::load("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json");
    println!("Llama vocab size: {}", llama_tokenizer.vocab.len());

    let mut safe_tensor_llama = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6");

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, &gpu);

    let mut kv_cache_llama = KVCache::new(llama_model.config.num_hidden_layers);

    let mut token_ids: Vec<usize> = vec![1];
    for i in 0..10 {
        let pos = token_ids.len() - 1;
        let last_id = *token_ids.last().unwrap();
        let output_id = llama_generate_token(LlamaGenerateTokenArgs {
            model: &llama_model,
            token_id: last_id,
            position: i,
            gpu: &gpu,
            kv_cache: &mut kv_cache_llama,
        });
        println!("pos={} id={}", i, output_id);
        token_ids.push(output_id)
    }



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
        let top_id = gpt2_generate_token(GPT2GenerateTokenArgs {
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
        let top_id = gpt2_generate_token(GPT2GenerateTokenArgs {
            token_id: last_id, position: pos,
            gpt2_model: &gpt2_model, kv_cache: &mut kv_cache, gpu: &gpu
        });
        token_ids.push(top_id);
    }

    let output: Vec<String> = token_ids.iter().map(|&id| gpt2_model.vocab.decode(id)).collect();
    let text = output.join("");
    println!("{}", text);
}
