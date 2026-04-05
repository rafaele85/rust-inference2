use crate::root::tokenizer::deserialize_eos_token_id;
use std::fs;
use serde::{Deserialize, Deserializer};
use serde_json::{from_str};
use crate::root::config::{ModelDimConfig, MoeConfig};

#[derive(Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default)]
    pub kv_dim: usize,
    #[serde(default)]
    pub heads_to_kvheads: usize,
    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
    #[serde(default)]
    pub attn_dim: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
}

impl LlamaConfig {
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(&path).expect(&format!("Unable to read file {}", &path));
        let mut config: LlamaConfig = from_str(&content).expect(&format!("Unable to deserialize file"));

        if config.head_dim == 0 {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }

        config.kv_dim = config.num_key_value_heads * config.head_dim;
        config.heads_to_kvheads = config.num_attention_heads / config.num_key_value_heads;
        config.attn_dim = config.num_attention_heads * config.head_dim;

        config
    }
}

impl MoeConfig for LlamaConfig {
    fn get_num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    fn get_moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
    }

    fn get_norm_topk_prob(&self) -> bool {
        false
    }

    fn get_routed_scaling_factor(&self) -> f32 {
        1.0
    }

    fn get_n_shared_experts(&self) -> usize {
        0
    }
}

impl ModelDimConfig for LlamaConfig {
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }    
}

