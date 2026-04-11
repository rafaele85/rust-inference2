use std::fs;
use crate::root::tokenizer::deserialize_eos_token_id;
use serde::Deserialize;
use serde_json::from_str;
use crate::root::config::{AttnConfig, MLAConfig, ModelDimConfig, MoeConfig, NormConfig};

#[derive(Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    #[serde(default)]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub mrope_interleaved: bool,
}

#[derive(Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,

    pub head_dim: usize,
    pub vocab_size: usize,

    pub rms_norm_eps: f32,

    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,

    pub shared_expert_intermediate_size: usize,
    pub rope_parameters: RopeParameters,
    pub layer_types: Vec<String>,

    // Linear attention specific
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_value_heads: usize,

    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,

    #[serde(default)]
    pub full_attention_interval: usize,
}

fn default_conv_kernel() -> usize { 4 }

#[derive(Deserialize)]
pub struct QuantizationConfig {
    #[serde(default = "default_bits")]
    pub bits: usize,
    #[serde(default = "default_group_size")]
    pub group_size: usize,
}

fn default_bits() -> usize { 4 }
fn default_group_size() -> usize { 64 }
#[derive(Deserialize)]
pub struct Qwen35Config {
    pub text_config: TextConfig,

    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl Qwen35Config {
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(&path).expect(&format!("Unable to read file {}", &path));
        let config: Qwen35Config = from_str(&content).expect(&format!("Unable to deserialize file"));
        config
    }

    pub fn is_linear_attention(&self, layer_no: usize) -> bool {
        self.text_config.layer_types[layer_no] == "linear_attention"
    }

    pub fn bits(&self) -> usize {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }

    pub fn group_size(&self) -> usize {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }
}

impl ModelDimConfig for Qwen35Config {
    fn get_hidden_size(&self) -> usize { self.text_config.hidden_size }
    fn get_num_hidden_layers(&self) -> usize { self.text_config.num_hidden_layers }
    fn get_vocab_size(&self) -> usize { self.text_config.vocab_size }
}

impl NormConfig for Qwen35Config {
    fn get_rms_norm_eps(&self) -> f32 { self.text_config.rms_norm_eps }
}

impl MoeConfig for Qwen35Config {
    fn get_num_experts_per_tok(&self) -> usize { self.text_config.num_experts_per_tok }
    fn get_moe_intermediate_size(&self) -> usize { self.text_config.moe_intermediate_size }
    fn get_norm_topk_prob(&self) -> bool { false }
    fn get_routed_scaling_factor(&self) -> f32 { 1.0 }
    fn get_n_shared_experts(&self) -> usize { 1 }
}

impl AttnConfig for TextConfig {
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn get_head_dim(&self) -> usize {
        self.head_dim
    }
}