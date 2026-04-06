use crate::root::arch::gpt2::gpt2::GPT2Model;
use crate::root::arch::safetensors::SafeTensor;
use crate::root::generate::{generate, GenerateArgs};
use crate::root::gpu::Gpu;
use crate::root::tokenizer::BpeTokenizer;

pub fn gpt2_tests(gpus: &mut Vec<Gpu>, max_tokens: usize) {

    let gpt2_base_path = "model";
    let gpt2_tokenizer = BpeTokenizer::load(
        gpt2_base_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );
    let mut safe_tensor_gpt2 = SafeTensor::new(gpt2_base_path);
    let gpt2_model = GPT2Model::new(&mut safe_tensor_gpt2, gpus);

    let prompt = "The capital of France is";

    let result = generate(GenerateArgs {
        prompt, tokenizer: &gpt2_tokenizer, max_tokens, gpus, eos_token_id: &gpt2_model.config.eos_token_id, num_layers: gpt2_model.config.n_layer
    }, &gpt2_model);
    println!("GPT-2: {}", result);

    let expected = "The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French Republic.\n\nThe French Republic is the capital of the French Republic.\n\nThe French Republic is the capital of the French Republic.\n\nThe";
    assert_eq!(result.trim(), expected.trim(), "GPT-2 output mismatch");
}