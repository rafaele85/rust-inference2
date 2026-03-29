use serde::Deserialize;

#[derive(Deserialize)]
pub struct DeepseekConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,  //dense FFN size (layer 0)
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rope_theta: f32,

    pub kv_lora_rank: usize,          //KV compression bottleneck
    pub qk_nope_head_dim: usize,      //Q/K dims without ROPE
    pub qk_rope_head_dim: usize,      //Q/K dims with ROPE
    pub v_head_dim: usize,            //value head dim
    #[serde(default)]
    pub q_lora_rank: Option<usize>,   //

    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize, //expert FFN size
    pub first_k_dense_replace: usize, //how many layers use dense FFN
    pub routed_scaling_factor: f32,   //
    pub scoring_func: String,         //"softmax"
    #[serde(default)]
    pub norm_topk_prob: bool,

    pub vocab_size: usize,
    #[serde(default)]
    pub eos_token_id: Vec<u32>,
}