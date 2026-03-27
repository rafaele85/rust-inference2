use crate::arch::llama::llama::LlamaModel;
use crate::arch::quant::awq::{awq_dequantize, AwqDequantizeArgs};
use crate::arch::safetensors::{load_tensor_f16, load_tensor_i32, resolve_hf_path, SafeTensor};
use crate::generate::{generate, GenerateArgs};
use crate::gpu::Gpu;
use crate::tokenizer::BpeTokenizer;

pub fn qwen_tests(gpus: &Vec<Gpu>) {
    let qwen_path = resolve_hf_path("Qwen/Qwen2.5-14B-Instruct");

    let qwen_tokenizer = BpeTokenizer::load(
        &qwen_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
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
        eos_token_id: &qwen_model.config.eos_token_id,
        num_layers: qwen_model.config.num_hidden_layers,
    }, &qwen_model);

    println!("Qwen: {}", result);
}


pub fn qwen3_tests(gpus: &Vec<Gpu>) {
    let qwen3_path = resolve_hf_path("Qwen/Qwen3-0.6B");

    let qwen3_tokenizer = BpeTokenizer::load(
        &qwen3_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
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
        eos_token_id: &qwen3_model.config.eos_token_id,
        num_layers: qwen3_model.config.num_hidden_layers,
    }, &qwen3_model);

    println!("Qwen3: {}", result);
}


pub fn qwen3_moe_tests(gpus: &Vec<Gpu>) {
    let qwen3_moe_path = resolve_hf_path("Qwen/Qwen3-30B-A3B-Instruct-2507");

    let qwen3_moe_tokenizer = BpeTokenizer::load(
        &qwen3_moe_path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );

    let mut qwen3_moe_safe_tensor = SafeTensor::new(&qwen3_moe_path);

    let qwen3_moe_model = LlamaModel::new(&mut qwen3_moe_safe_tensor, &gpus);

    for (i, gpu) in gpus.iter().enumerate() {
        let (free, total) = gpu.mem_info();
        let used = total - free;
    }

    let prompt = "The capital of France is";
    //let prompt = "<|im_start|>user\nExplain the difference between a compiler and an interpreter in simple terms.<|im_end|>\n<|im_start|>assistant\n";
    //let prompt = "Solve step by step: A train leaves Station A at 9:00 AM traveling east at 80 km/h. Another train leaves Station B, 400 km east of A, at 10:00 AM traveling west at 120 km/h. At what time do they meet and how far from Station A?";

    let result = generate(GenerateArgs {
        prompt,
        tokenizer: &qwen3_moe_tokenizer,
        max_tokens: 50,
        gpus: &gpus,
        eos_token_id: &qwen3_moe_model.config.eos_token_id,
        num_layers: qwen3_moe_model.config.num_hidden_layers,
    }, &qwen3_moe_model);

    println!("Qwen3-MoE: {}", result);
}

pub fn qwen_awq_tests() {
    let qwen_awq_path = resolve_hf_path("Qwen/Qwen2.5-0.5B-Instruct-AWQ");
    let mut st = SafeTensor::new(&qwen_awq_path);

    let qw = load_tensor_i32(&mut st, "model.layers.0.self_attn.q_proj.qweight");
    let qz = load_tensor_i32(&mut st, "model.layers.0.self_attn.q_proj.qzeros");
    let sc = load_tensor_f16(&mut st, "model.layers.0.self_attn.q_proj.scales");

    let w = awq_dequantize(AwqDequantizeArgs {qweight: &qw, qzero: &qz, scales: &sc, in_dim: 896, out_dim: 896, group_size: 128});

    println!("qweight first 10:");
    for v in &qw[..10] {
        println!(" {}", v)
    }

    println!("scales first 10:");
    for v in &sc[..10] {
        println!(" {}", v)
    }

    println!("Row 895, first 10:");
    for i in 0..10 {
        println!(" {}", w[895*896 + i])
    }

    print!("col 0, rows 0..10:");
    for row in 0..10 { print!(" {}", w[row * 896 + 0]); }
    println!();

    // y[:10, 0] = W[0, 0..10] = row 0, first 10 cols
    print!("row 0, cols 0..10:");
    for col in 0..10 { print!(" {}", w[0 * 896 + col]); }
    println!();

    print!("in=0, out=0..10:");
    for out_idx in 0..10 { print!(" {}", w[out_idx * 896 + 0]); }
    println!();

}
