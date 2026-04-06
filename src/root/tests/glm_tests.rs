use crate::root::arch::glm::glm::GLMModel;
use crate::root::arch::safetensors::{load_tensor_to_f32, resolve_hf_path, SafeTensor};
use crate::root::gpu::Gpu;
use crate::root::tokenizer::BpeTokenizer;
use ndarray_npy::NpzReader;
use std::fs::File;
use crate::root::arch::common::LoadLayer;
use crate::root::arch::glm::glm_config::GLMConfig;
use crate::root::arch::glm::glm_layer::GLMLayer;
use crate::root::common_types::Vecf32;
use crate::root::generate::{generate, GenerateArgs};
use crate::root::matmul::matmul_transposed_f16;
use crate::root::norm::rms_norm;

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

    println!("Generatring");
    let result = generate(GenerateArgs {
        prompt, tokenizer: &tokenizer, max_tokens, gpus,
        eos_token_id: &model.config.eos_token_id,
        num_layers: model.config.num_hidden_layers
    }, &model);
    println!("GLM: {}", result);
}

pub fn glm_test_config(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let file = File::open("load_test_reference.npz").unwrap();
    let mut npz: NpzReader<File> = NpzReader::new(file).unwrap();

    let base_path = resolve_hf_path("cyankiwi/GLM-4.7-Flash-AWQ-4bit");
    let mut safe_tensor = SafeTensor::new(&base_path);
    let model = GLMModel::new(&mut safe_tensor, gpus);
    let config = &model.config;

    // numpy defaults: int=int64, float=float64
    let hidden: ndarray::ArrayD<i64> = npz.by_name("hidden_size").unwrap();
    let hidden_val = hidden.iter().next().unwrap();
    assert_eq!(config.hidden_size, *hidden_val as usize);

    let layers: ndarray::ArrayD<i64> = npz.by_name("num_hidden_layers").unwrap();
    let layers_val = layers.iter().next().unwrap();
    assert_eq!(config.num_hidden_layers, *layers_val as usize);

    let q_rank: ndarray::ArrayD<i64> = npz.by_name("q_lora_rank").unwrap();
    let q_rank_val = q_rank.iter().next().unwrap();
    assert_eq!(config.q_lora_rank, *q_rank_val as usize);

    let eps: ndarray::ArrayD<f64> = npz.by_name("rms_norm_eps").unwrap();
    let eps_val = eps.iter().next().unwrap();
    assert!((config.rms_norm_eps - *eps_val as f32).abs() < 1e-6);

    println!("✅ Config loading correct");
}

pub fn glm_test_weights(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let base_path = resolve_hf_path("cyankiwi/GLM-4.7-Flash-AWQ-4bit");
    let mut safe_tensor = SafeTensor::new(&base_path);
    let model = GLMModel::new(&mut safe_tensor, gpus);

    let file = File::open("load_test_reference.npz").unwrap();
    let mut npz: NpzReader<File> = NpzReader::new(file).unwrap();

    // Test embeddings
    let embed = load_tensor_to_f32(&mut safe_tensor, "model.embed_tokens.weight");
    let expected_embed: ndarray::Array1<f32> = npz.by_name("embed_sample").unwrap();

    for (i, (got, exp)) in embed.iter().zip(expected_embed.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-2, "embed[{}]: got {}, expected {}", i, got, exp);
    }
    println!("✅ Embeddings sample correct");

    // Test final norm
    let final_norm = load_tensor_to_f32(&mut safe_tensor, "model.norm.weight");
    let expected_norm: ndarray::Array1<f32> = npz.by_name("final_norm_sample").unwrap();

    for (i, (got, exp)) in final_norm.iter().zip(expected_norm.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-2, "final_norm[{}]: got {}, expected {}", i, got, exp);
    }
    println!("✅ Final norm sample correct");

    // Test layer 0 q_a_proj shape - numpy uses int64 by default
    let q_a = load_tensor_to_f32(&mut safe_tensor, "model.layers.0.self_attn.q_a_proj.weight");
    let expected_shape: ndarray::Array1<i64> = npz.by_name("q_a_shape").unwrap();  // i64 not i32
    let expected_len = (expected_shape[0] * expected_shape[1]) as usize;
    assert_eq!(q_a.len(), expected_len, "q_a_proj length mismatch: got {}, expected {}", q_a.len(), expected_len);

    // Verify first 10 elements
    let expected_q_a: ndarray::Array1<f32> = npz.by_name("q_a_sample").unwrap();
    for (i, (got, exp)) in q_a.iter().zip(expected_q_a.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-2, "q_a[{}]: got {}, expected {}", i, got, exp);
    }
    println!("✅ Layer 0 q_a_proj sample correct");
}



pub fn glm_attention_step2_kv_proj(
    hidden: &Vecf32,  // [2048]
    layer: &GLMLayer,
    config: &GLMConfig,
    gpu: &mut Gpu,
) -> (Vecf32, Vecf32, Vecf32) {  // (keys_nope [3840], values [5120], k_pe [64])

    // 1. Compress: [2048] -> [576] (512 c_cache + 64 rope)
    let kv_latent = matmul_transposed_f16(hidden, &layer.kv_a_proj_w, 576, gpu);

    // 2. Split: c_cache [512], k_pe [64]
    let (c_cache, k_pe) = kv_latent.split_at(config.kv_lora_rank);  // 512 and 64

    // 3. RMSNorm on c_cache [512]
    let c_cache_norm = rms_norm(c_cache, &layer.kv_a_layernorm_w, config.rms_norm_eps);

    // 4. Expand: [512] -> [8960] (20 heads * (192+256))
    let kv_full = matmul_transposed_f16(&c_cache_norm, &layer.kv_b_proj_w, 8960, gpu);

    // 5. Split into keys (no-rope) and values
    let kv_head_dim = config.qk_nope_head_dim + config.v_head_dim;  // 192 + 256 = 448
    let total_kv_dim = config.num_attention_heads * kv_head_dim;    // 20 * 448 = 8960

    let keys_nope_len = config.num_attention_heads * config.qk_nope_head_dim;  // 3840
    let values_len = config.num_attention_heads * config.v_head_dim;           // 5120

    let keys_nope = kv_full[..keys_nope_len].to_vec();  // [3840] = [20, 192]
    let values = kv_full[keys_nope_len..].to_vec();     // [5120] = [20, 256]

    (keys_nope, values, k_pe.to_vec())
}
pub fn glm_test_step2(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let file = File::open("step2_kv_proj_reference.npz").unwrap();
    let mut npz: NpzReader<File> = NpzReader::new(file).unwrap();

    let hidden: ndarray::Array1<f32> = npz.by_name("input").unwrap();

    // Load layer 0
    let base_path = resolve_hf_path("cyankiwi/GLM-4.7-Flash-AWQ-4bit");
    let mut safe_tensor = SafeTensor::new(&base_path);
    let model = GLMModel::new(&mut safe_tensor, gpus);

    let layer0 = GLMLayer::load_layer(&mut safe_tensor, 0, &mut gpus[0]);

    let (keys, values, k_pe) = glm_attention_step2_kv_proj(&hidden.to_vec(), &layer0, &model.config, &mut gpus[0]);

    assert_eq!(keys.len(), 3840);
    assert_eq!(values.len(), 5120);
    assert_eq!(k_pe.len(), 64);

    // Verify values
    let expected_keys: ndarray::Array1<f32> = npz.by_name("keys_nope").unwrap();
    let max_diff: f32 = keys.iter().zip(expected_keys.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    println!("Keys max diff: {}", max_diff);
    assert!(max_diff < 1e-2);

    println!("✅ Step 2 KV projection correct");
}