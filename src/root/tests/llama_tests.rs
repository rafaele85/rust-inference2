use crate::root::arch::llama::llama::LlamaModel;
use crate::root::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::root::generate::{generate, GenerateArgs};
use crate::root::gpu::Gpu;
use crate::root::tokenizer::BpeTokenizer;

pub fn llama_tests(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let llama_base_path = &resolve_hf_path("TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    let llama_tokenizer = BpeTokenizer::load(
        llama_base_path,
        "\u{2581}".to_string(),
        "\u{010A}".to_string(),
        true
    );

    let mut safe_tensor_llama = SafeTensor::new(llama_base_path);

    let llama_model = LlamaModel::new(&mut safe_tensor_llama, gpus);

    let prompt = "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n";
    let result = generate(GenerateArgs {
        prompt, tokenizer: &llama_tokenizer, max_tokens, gpus, eos_token_id: &llama_model.config.eos_token_id, num_layers: llama_model.config.num_hidden_layers
    }, &llama_model);
    println!("LLama: {}", result);

    let expected = "<|system|>\nYou are a helpful assistant.</s> <|user|>\nWhat is the capital of France?</s> <|assistant|>\nThe capital of France is Paris.</s>";
    assert_eq!(result.trim(), expected.trim(), "LLama output mismatch");

}