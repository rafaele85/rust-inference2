use crate::root::arch::common::load_all_layers;
use crate::root::arch::glm::glm_config::GLMConfig;
use crate::root::arch::glm::glm_layer::GLMLayer;
use crate::root::arch::glm::glm_transformer::{glm_generate_token, GLMGenerateTokenArgs};
use crate::root::arch::safetensors::{load_tensor_to_f32, SafeTensor};
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::generate::GenerateNextTokenArgs;
use crate::root::gpu::{get_default_gpu, Gpu};
use crate::root::model::{GenerateNextToken, Model};

pub struct GLMModel {
    pub layers: Vec<GLMLayer>,
    pub config: GLMConfig,
    pub embed_weights: Vecf32,
    pub final_norm_w: Vecf32,
    pub lm_head_w: WeightF16,
}

impl GLMModel {
    pub fn new(safe_tensor: &mut SafeTensor, gpus: &mut Vec<Gpu>) -> Self {

        let config = GLMConfig::load(&format!("{}/config.json", safe_tensor.path_to_model_dir));
        let layers = load_all_layers(safe_tensor, config.num_hidden_layers, gpus);

        let embed_weights = load_tensor_to_f32(safe_tensor, "model.embed_tokens.weight");
        let final_norm_w = load_tensor_to_f32(safe_tensor, "model.norm.weight");

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



impl GenerateNextToken for GLMModel {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize {
        let GenerateNextTokenArgs { token_id, position, kv_cache, gpus} = args;
        let res = glm_generate_token(GLMGenerateTokenArgs {token_id, position, kv_cache, gpus, model: self});
        res
    }
}


impl Model for GLMModel {
    fn get_embed_weights(&self) -> &Vecf32 {
        &self.embed_weights
    }

    fn get_final_norm_w(&self) -> &Vecf32 {
        &self.final_norm_w
    }

    fn get_lm_head_w(&self) -> &WeightF16 {
        &self.lm_head_w
    }
}


