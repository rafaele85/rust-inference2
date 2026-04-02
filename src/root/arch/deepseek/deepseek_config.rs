use crate::root::tokenizer::deserialize_eos_token_id;
use std::fs;
use serde::Deserialize;
use serde_json::from_str;

#[derive(Deserialize)]
pub struct DeepseekConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,  //dense FFN size (layer 0)
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,              //base frequency for RoPE

    pub kv_lora_rank: usize,          //KV compression bottleneck
    pub qk_nope_head_dim: usize,      //Q/K dims without ROPE
    pub qk_rope_head_dim: usize,      //Q/K dims with ROPE
    pub v_head_dim: usize,            //value head dim
    #[serde(default)]
    pub q_lora_rank: Option<usize>,   //bottleneck size for compressing Q, if > 0 means Q goes through down-project→norm→up-projec

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