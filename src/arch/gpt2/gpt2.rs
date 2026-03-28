use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;
use crate::arch::common::{load_all_layers};
use crate::arch::safetensors::{load_tensor_f32, SafeTensor};
use crate::arch::gpt2::gpt2_config::GPT2Config;
use crate::arch::gpt2::gpt2_layer::{GPT2Layer};
use crate::arch::gpt2::gpt2_transformer::{gpt2_generate_token, GPT2GenerateTokenArgs};
use crate::generate::{GenerateNextTokenArgs, Model};


pub struct GPT2Model {
    pub wte: Vecf32,
    pub wte_gpu: Weight,
    pub wpe: Vecf32,
    pub ln_f_w: Vecf32,
    pub ln_f_b: Vecf32,
    pub layers: Vec<GPT2Layer>,
    pub config: GPT2Config,
}

impl GPT2Model {
    pub fn new(safe_tensors: &mut SafeTensor, gpus: &mut Vec<Gpu>) -> Self {
        let config = GPT2Config::load(&format!("{}/config.json", safe_tensors.path_to_model_dir));
        let wte = load_tensor_f32(safe_tensors, "wte.weight");
        let wte_gpu = gpus[0].upload(&wte, "wte_gpu");
        let wpe = load_tensor_f32(safe_tensors, "wpe.weight");
        let ln_f_w = load_tensor_f32(safe_tensors, "ln_f.weight");
        let ln_f_b = load_tensor_f32(safe_tensors, "ln_f.bias");
        let layers = load_all_layers(safe_tensors, config.n_layer, gpus);


        Self {wte, wte_gpu, wpe, ln_f_w, ln_f_b, layers, config}
    }
}

impl Model for GPT2Model {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize {
        let GenerateNextTokenArgs { token_id, position, kv_cache, gpus} = args;
        let res = gpt2_generate_token(GPT2GenerateTokenArgs {token_id, position, kv_cache, gpus, model: self});
        res
    }
}