use crate::arch::llama::llama::LlamaModel;
use crate::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::generate::{generate, GenerateArgs};
use crate::gpu::Gpu;
use crate::tokenizer::BpeTokenizer;

pub fn llama_tests(gpus: &Vec<Gpu>) {
    let llama_base_path = &resolve_hf_path("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    let llama_tokenizer = BpeTokenizer::load(
        llama_base_path,
        "\u{2581}".to_string(),
        true
    );

    let mut safe_tensor_llama = SafeTensor::new(llama_base_path);

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, &gpus);

    let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    let max_tokens = 200;
    let result = generate(GenerateArgs {
        prompt, tokenizer: &llama_tokenizer, max_tokens, gpus: &gpus, eos_token_id: llama_model.config.eos_token_id, num_layers: llama_model.config.num_hidden_layers
    }, &llama_model);
    println!("LLama: {}", result);
}