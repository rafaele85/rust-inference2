pub trait MLAConfig {
    fn get_v_head_dim(&self) -> usize;
    fn get_rope_theta(&self) -> f32;
    fn get_kv_lora_rank(&self) -> usize;
    fn get_qk_rope_head_dim(&self) -> usize;
    fn get_qk_nope_head_dim(&self) -> usize;
}

pub trait AttnConfig {
    fn get_num_attention_heads(&self) -> usize;
    fn get_head_dim(&self) -> usize;
}

pub trait NormConfig {
    fn get_rms_norm_eps(&self) -> f32;
}

pub trait ModelDimConfig {
    fn get_hidden_size(&self) -> usize;
    fn get_num_hidden_layers(&self) -> usize;
    fn get_vocab_size(&self) -> usize;
}


pub trait MoeConfig {
    fn get_num_experts_per_tok(&self) -> usize;
    fn get_moe_intermediate_size(&self) -> usize;
    fn get_norm_topk_prob(&self) -> bool;
    fn get_routed_scaling_factor(&self) -> f32;
    fn get_n_shared_experts(&self) -> usize;
}

pub trait FFNConfig {
    fn get_intermediate_size(&self) -> usize;
}

pub trait MLXConfig {
    fn get_bits(&self) -> usize;
    fn get_group_size(&self) -> usize;
}