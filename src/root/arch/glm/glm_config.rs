use std::fs;
use serde_json::from_str;
use crate::root::tokenizer::deserialize_eos_token_id;
use serde::Deserialize;
use crate::root::config::{FFNConfig, MLAConfig, ModelDimConfig, MoeConfig, NormConfig};

#[derive(Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

#[derive(Deserialize)]
pub struct GLMConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub rms_norm_eps: f32,
    pub num_hidden_layers: usize,
    pub v_head_dim: usize,
    pub rope_parameters: RopeParameters,
    pub vocab_size: usize,
    pub intermediate_size: usize,
    // MoE
    #[serde(default)]
    pub n_routed_experts: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub routed_scaling_factor: f32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

impl GLMConfig {
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(&path).expect(&format!("Unable to read file {}", &path));
        let config: GLMConfig = from_str(&content).expect(&format!("Unable to deserialize file"));

        config
    }
}

impl NormConfig for GLMConfig {
    fn get_rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
}





impl ModelDimConfig for GLMConfig {
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn get_num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}


impl MLAConfig for GLMConfig{
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn get_v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn get_rope_theta(&self) -> f32 { self.rope_parameters.rope_theta }
    fn get_kv_lora_rank(&self) -> usize {
        self.kv_lora_rank
    }
    fn get_qk_rope_head_dim(&self) -> usize {
        self.qk_rope_head_dim
    }
    fn get_qk_nope_head_dim(&self) -> usize {
        self.qk_nope_head_dim
    }
}

impl FFNConfig for GLMConfig {
    fn get_intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

impl MoeConfig for GLMConfig {
    fn get_norm_topk_prob(&self) -> bool {
        self.norm_topk_prob
    }

    fn get_moe_intermediate_size(&self) -> usize {
        self.moe_intermediate_size
    }

    fn get_num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    fn get_routed_scaling_factor(&self) -> f32 {
        self.routed_scaling_factor
    }

    fn get_n_shared_experts(&self) -> usize {
        self.n_shared_experts
    }
}