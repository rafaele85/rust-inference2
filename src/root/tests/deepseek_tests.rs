use crate::root::arch::deepseek::deepseek::DeepseekModel;
use crate::root::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::root::generate::{generate, GenerateArgs, GenerateNextTokenArgs};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::KVCache;
use crate::root::model::GenerateNextToken;
use crate::root::tokenizer::BpeTokenizer;

pub fn deepseek_tests(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let base_path = &resolve_hf_path("TechxGenus/DeepSeek-V2-Lite-Chat-AWQ");
    let tokenizer = BpeTokenizer::load(
        base_path,
        "\u{0120}".to_string(),  // Ġ (represents space in BBPE)
        "\u{010A}".to_string(),  // Ċ (represents newline in BBPE)
        false
    );

    let mut safe_tensor = SafeTensor::new(base_path);

    let model = DeepseekModel::new(&mut safe_tensor, gpus);

    let prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:";

    let result = generate(GenerateArgs {
        prompt, tokenizer: &tokenizer, max_tokens, gpus, eos_token_id: &model.config.eos_token_id, num_layers: model.config.num_hidden_layers
    }, &model);
    println!("Deepseek: {}", result);

    let expected = "User: What is the capital of France?\n\nAssistant: The capital of France is Paris.";
    assert_eq!(result.trim(), expected.trim(), "Deepseek output mismatch");
}
