use crate::root::arch::deepseek::deepseek::DeepseekModel;
use crate::root::arch::safetensors::{resolve_hf_path, SafeTensor};
use crate::root::generate::{generate, GenerateArgs, GenerateNextTokenArgs, Model};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::KVCache;
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
}

pub fn deepseek_single_token_test(gpus: &mut Vec<Gpu>) {
    let base_path = &resolve_hf_path("TechxGenus/DeepSeek-V2-Lite-Chat-AWQ");
    let tokenizer = BpeTokenizer::load(
        base_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );

    let mut safe_tensor = SafeTensor::new(base_path);
    let model = DeepseekModel::new(&mut safe_tensor, gpus);

    let prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:";
    let input_ids = tokenizer.encode(prompt);
    println!("First token ID: {}", input_ids[0]);

    let mut kv_cache = KVCache::new(model.config.num_hidden_layers);
    let token_id = model.generate_next_token(GenerateNextTokenArgs {
        token_id: input_ids[0] as usize,
        position: 0,
        kv_cache: &mut kv_cache,
        gpus,
    });

    println!("Output token: {} -> {:?}", token_id, tokenizer.id_to_token.get(&(token_id as u32)));
}


pub fn deepseek_multi_token_test(gpus: &mut Vec<Gpu>) {
    let base_path = &resolve_hf_path("TechxGenus/DeepSeek-V2-Lite-Chat-AWQ");
    let tokenizer = BpeTokenizer::load(
        base_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );

    let mut safe_tensor = SafeTensor::new(base_path);
    let model = DeepseekModel::new(&mut safe_tensor, gpus);

    let prompt = "<｜begin▁of▁sentence｜>User: What is the capital of France?\n\nAssistant:";
    let input_ids = tokenizer.encode(prompt);
    println!("Token IDs ({}):", input_ids.len());

    let mut kv_cache = KVCache::new(model.config.num_hidden_layers);

    for i in 0..input_ids.len() {
        let token_id = model.generate_next_token(GenerateNextTokenArgs {
            token_id: input_ids[i] as usize,
            position: i,
            kv_cache: &mut kv_cache,
            gpus,
        });
        println!("pos={} token_in={} token_out={}", i, input_ids[i], token_id);
    }
}