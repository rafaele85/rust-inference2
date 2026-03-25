use std::io::{Read, Seek};
use std::io::SeekFrom::Start;
use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;
use crate::arch::common::{load_all_layers};
use crate::arch::llama::llama_config::LlamaConfig;
use crate::arch::llama::llama_layer::LlamaLayer;
use crate::arch::llama::llama_transformer::{llama_generate_token, LlamaGenerateTokenArgs};
use crate::arch::safetensors::SafeTensor;
use crate::generate::{GenerateNextTokenArgs, Model};

pub fn load_tensor_bf16(safe_tensor: &mut SafeTensor, name: &str) -> Vecf32 {
    if !safe_tensor.entries.contains_key(name) {
        panic!("NOT FOUND KEY {}", name);
    }

    let entry = &safe_tensor.entries[name];
    let shard_idx = entry.shard_idx;
    let start = safe_tensor.data_offsets[shard_idx] + entry.offsets[0];
    let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;

    let mut bytes = vec![0u8; byte_len];
    safe_tensor.files[shard_idx].seek(Start(start)).unwrap();
    safe_tensor.files[shard_idx].read_exact(&mut bytes).expect("Failed to read file");

    let result = bytes.chunks_exact(2).map(|c| {
        let bits = u16::from_le_bytes(c.try_into().unwrap());
        f32::from_bits((bits as u32) << 16)
    }).collect();

    result
}



pub struct LlamaModel {
    pub layers: Vec<LlamaLayer>,
    pub config: LlamaConfig,
    pub embed_weights: Vecf32,
    pub final_norm_w: Vecf32,
    pub lm_head_w: Weight,
}

impl LlamaModel {
    pub fn new(safe_tensor: &mut SafeTensor, gpu: &Gpu) -> Self {
        let config = LlamaConfig::load(&format!("{}/config.json", safe_tensor.path_to_model_dir));
        let layers = load_all_layers(safe_tensor, config.num_hidden_layers, &gpu);
        let embed_weights = load_tensor_bf16(safe_tensor, "model.embed_tokens.weight");
        let final_norm_w = load_tensor_bf16(safe_tensor,"model.norm.weight");
        let lm_head_w  = gpu.upload(&load_tensor_bf16(safe_tensor, "lm_head.weight"), "lm_head_w" );

        Self {layers, config, embed_weights, final_norm_w, lm_head_w}
    }
}


impl Model for LlamaModel {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize {
        let GenerateNextTokenArgs { token_id, position, kv_cache, gpu} = args;
        let res = llama_generate_token(LlamaGenerateTokenArgs {token_id, position, kv_cache, gpu, model: self});
        res
    }
}
