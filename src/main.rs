mod matmul;
mod activation;
mod kv_cache;
mod common_types;
mod gpu;
mod arch;
mod linear;
mod tokenizer;
mod generate;

use crate::gpu::Gpu;
use arch::safetensors::SafeTensor;
use crate::arch::gpt2::gpt2::GPT2Model;
use crate::arch::llama::llama::LlamaModel;
use crate::generate::{generate, GenerateArgs};
use crate::tokenizer::BpeTokenizer;

fn main() {
    let gpu = Gpu::new(0);


    let llama_tokenizer = BpeTokenizer::load(
        "/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6/tokenizer.json",
        "\u{2581}".to_string(),
        true
    );

    let mut safe_tensor_llama = SafeTensor::open("/home/rambo/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6");

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, &gpu);

    let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    let max_tokens = 200;
    let result = generate(GenerateArgs {
        prompt, tokenizer: &llama_tokenizer, max_tokens, gpu: &gpu, eos_token_id: llama_model.config.eos_token_id, num_layers: llama_model.config.num_hidden_layers
    }, &llama_model);
    println!("LLama: {}", result);


    //-----------GPT2-------------

    let gpt2_tokenizer = BpeTokenizer::load(
        "model/tokenizer.json",
        "\u{0120}".to_string(),
        false
    );
    let mut safe_tensor_gpt2 = SafeTensor::open("model");
    let gpt2_model = GPT2Model::new(&mut safe_tensor_gpt2, &gpu);

    let prompt = "The capital of France is";
    let max_tokens = 50;

    let result = generate(GenerateArgs {
        prompt, tokenizer: &gpt2_tokenizer, max_tokens, gpu: &gpu, eos_token_id: gpt2_model.config.eos_token_id, num_layers: gpt2_model.config.n_layer
    }, &gpt2_model);
    println!("GPT-2: {}", result);

}
