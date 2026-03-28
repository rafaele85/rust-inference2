use crate::arch::gpt2::gpt2::GPT2Model;
use crate::arch::safetensors::SafeTensor;
use crate::generate::{generate, GenerateArgs};
use crate::gpu::Gpu;
use crate::tokenizer::BpeTokenizer;

pub fn gpt2_tests(gpus: &mut Vec<Gpu>) {

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
    let max_tokens = 50;

    let result = generate(GenerateArgs {
        prompt, tokenizer: &gpt2_tokenizer, max_tokens, gpus, eos_token_id: &gpt2_model.config.eos_token_id, num_layers: gpt2_model.config.n_layer
    }, &gpt2_model);
    println!("GPT-2: {}", result);
}