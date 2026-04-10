use crate::root::tokenizer::deserialize_eos_token_id;
use std::fs;
use serde::Deserialize;
use serde_json::from_str;
use crate::root::config::{AttnConfig, FFNConfig, MLAConfig, MLXConfig, ModelDimConfig, MoeConfig, NormConfig};

#[derive(Deserialize)]
pub struct DeepseekConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,  //dense FFN size (layer 0)
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,

    pub num_attention_heads: usize,
    pub rope_theta: f32,              //base frequency for RoPE

    pub kv_lora_rank: usize,          //KV compression bottleneck
    pub qk_nope_head_dim: usize,      //Q/K dims without ROPE
    pub qk_rope_head_dim: usize,      //Q/K dims with ROPE
    pub v_head_dim: usize,            //value head dim

    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize, // FFN hidden size for each individual expert
    pub first_k_dense_replace: usize, // number of layers at the start that use dense FFN instead of MoE.
    pub routed_scaling_factor: f32,   //multiplier applied to the routed expert output (amplifies expert contribution)
    pub scoring_func: String,         //"softmax"
    #[serde(default)]
    pub norm_topk_prob: bool,         // whether to renormalize expert probabilities after top-k selection.  For sigmoid routing (GLM) it would actually matter

    pub vocab_size: usize,
    #[serde(deserialize_with = "deserialize_eos_token_id")]
    pub eos_token_id: Vec<u32>,
}

impl DeepseekConfig {
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(&path).expect(&format!("Unable to read file {}", &path));
        let config: DeepseekConfig = from_str(&content).expect(&format!("Unable to deserialize file"));

        config
    }
}

impl ModelDimConfig for DeepseekConfig {
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

impl NormConfig for DeepseekConfig {
    fn get_rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }
}


impl MLAConfig for DeepseekConfig{
    fn get_v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn get_rope_theta(&self) -> f32 {
        self.rope_theta
    }
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

impl FFNConfig for DeepseekConfig {
    fn get_intermediate_size(&self) -> usize {
        self.intermediate_size
    }
}

impl MoeConfig for DeepseekConfig {
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

impl MLXConfig for DeepseekConfig {
    fn get_bits(&self) -> usize {
        0
    }
    fn get_group_size(&self) -> usize {
        0
    }
}

impl AttnConfig for DeepseekConfig {
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    fn get_head_dim(&self) -> usize {
        0
    }
}