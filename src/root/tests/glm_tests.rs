use crate::root::arch::glm::glm::GLMModel;
use crate::root::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::root::gpu::Gpu;
use crate::root::tokenizer::BpeTokenizer;
use crate::root::generate::{generate, GenerateArgs};

pub fn glm_tests(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    println!("Loading model");
    let base_path = &resolve_hf_path("cyankiwi/GLM-4.7-Flash-AWQ-4bit");
    let tokenizer = BpeTokenizer::load(
        base_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );

    let mut safe_tensor = SafeTensor::new(base_path);
    let model = GLMModel::new(&mut safe_tensor, gpus);

    let prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:";

    println!("Generating");
    let result = generate(GenerateArgs {
        prompt, tokenizer: &tokenizer, max_tokens, gpus,
        eos_token_id: &model.config.eos_token_id,
        num_layers: model.config.num_hidden_layers
    }, &model);
    println!("GLM: {}", result);

    // glm_tests
    let expected = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant: The capital of France is Paris.\n\nUser: What is the capital of the United States?\n\nAssistant: The capital of the United States is Washington, D.C.\n\nUser: What is the capital of Italy?\n\nAssistant: The capital of Italy is Rome.\n\nUser";
    assert_eq!(result.trim(), expected.trim(), "GLM output mismatch");
}
