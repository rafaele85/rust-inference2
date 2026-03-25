use crate::arch::llama::llama::LlamaModel;
use crate::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::generate::{generate, GenerateArgs};
use crate::gpu::Gpu;
use crate::tokenizer::BpeTokenizer;

pub fn qwen_tests(gpus: &Vec<Gpu>) {
    let qwen_path = resolve_hf_path("Qwen/Qwen2.5-14B-Instruct");

    let qwen_tokenizer = BpeTokenizer::load(
        &qwen_path,
        "\u{0120}".to_string(),
        false
    );

    let mut safe_tensor_qwen = SafeTensor::new(&qwen_path);
    let qwen_model = LlamaModel::new(&mut safe_tensor_qwen, &gpus);

    let prompt = "The capital of France is";

    let result = generate(GenerateArgs {
        prompt,
        tokenizer: &qwen_tokenizer,
        max_tokens: 50,
        gpus: &gpus,
        eos_token_id: qwen_model.config.eos_token_id,
        num_layers: qwen_model.config.num_hidden_layers,
    }, &qwen_model);

    println!("Qwen: {}", result);
}


pub fn qwen3_tests(gpus: &Vec<Gpu>) {
    let qwen3_path = resolve_hf_path("Qwen/Qwen3-0.6B");

    let qwen3_tokenizer = BpeTokenizer::load(
        &qwen3_path,
        "\u{0120}".to_string(),
        false
    );

    let mut qwen3_safe_tensor = SafeTensor::new(&qwen3_path);
    let qwen3_model = LlamaModel::new(&mut qwen3_safe_tensor, &gpus);

    let prompt = "The capital of France is";

    let result = generate(GenerateArgs {
        prompt,
        tokenizer: &qwen3_tokenizer,
        max_tokens: 50,
        gpus: &gpus,
        eos_token_id: qwen3_model.config.eos_token_id,
        num_layers: qwen3_model.config.num_hidden_layers,
    }, &qwen3_model);

    println!("Qwen3: {}", result);
}