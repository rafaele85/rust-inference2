use crate::root::tokenizer::deserialize_eos_token_id;
use std::fs::read_to_string;
use serde::Deserialize;
use serde_json::from_str;
use crate::root::config::ModelDimConfig;

#[derive(Deserialize)]
pub struct GPT2Config {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub vocab_size: usize,
    pub layer_norm_epsilon: f32,
    #[serde(default)]
    pub ffn_dim: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

impl GPT2Config {
    pub fn load(path: &str) -> Self {
        let content = read_to_string(path).expect(&format!("Failed to read config file {}", &path));
        let mut config: GPT2Config = from_str(&content).expect("Failed to parse config file");
        config.ffn_dim = config.n_embd * 4;
        config.head_dim = config.n_embd / config.n_head;

        config
    }
}

impl ModelDimConfig for GPT2Config {
    fn get_num_hidden_layers(&self) -> usize {
        self.n_layer
    }

    fn get_hidden_size(&self) -> usize {
        self.n_embd
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}
