use crate::root::arch::common::load_all_layers;
use crate::root::arch::deepseek::deepseek_config::DeepseekConfig;
use crate::root::arch::deepseek::deepseek_layer::DeepseekLayer;
use crate::root::arch::deepseek::deepseek_transformer::{deepseek_generate_token, DeepseekGenerateTokenArgs};
use crate::root::arch::safetensors::{load_tensor_to_f32, SafeTensor};
use crate::root::common_types::{Vecf16, Vecf32, WeightF16};
use crate::root::generate::{GenerateNextTokenArgs, Model};
use crate::root::gpu::{get_default_gpu, Gpu};

pub struct DeepseekModel {
    pub layers: Vec<DeepseekLayer>,
    pub config: DeepseekConfig,
    pub embed_weights: Vecf32,
    pub final_norm_w: Vecf32,
    pub lm_head_w: WeightF16
}

impl DeepseekModel {
    pub fn new(safe_tensor: &mut SafeTensor, gpus: &mut Vec<Gpu>) -> Self {

        let config = DeepseekConfig::load(&format!("{}/config.json", safe_tensor.path_to_model_dir));

        let layers = load_all_layers(safe_tensor, config.num_hidden_layers, gpus);
        let embed_weights = load_tensor_to_f32(safe_tensor, "model.embed_tokens.weight");
        let final_norm_w = load_tensor_to_f32(safe_tensor,"model.norm.weight");
        let default_gpu = get_default_gpu(gpus, "lm_head_w");

        let lm_head_exists = safe_tensor.entries.contains_key("lm_head.weight");

        let lm_head_w  = if lm_head_exists {
            let raw = &load_tensor_to_f32(safe_tensor, "lm_head.weight");
            default_gpu.upload_f16_from_f32(raw, "lm_head_w" )
        } else {
            default_gpu.upload_f16_from_f32(&embed_weights, "lm_head_w" )
        };

        Self {layers, config, embed_weights, final_norm_w, lm_head_w}
    }
}



impl Model for DeepseekModel {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize {
        let GenerateNextTokenArgs { token_id, position, kv_cache, gpus} = args;
        let res = deepseek_generate_token(DeepseekGenerateTokenArgs {token_id, position, kv_cache, gpus, model: self});
        res
    }
}
