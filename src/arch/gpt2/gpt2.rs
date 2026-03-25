use std::io::SeekFrom::Start;
use crate::common_types::{Vecf32, Weight};
use crate::gpu::Gpu;
use crate::arch::common::{load_all_layers, LoadLayer};
use crate::arch::safetensors::SafeTensor;
use std::io::{Read, Seek};
use crate::arch::gpt2::gpt2_config::GPT2Config;
use crate::arch::gpt2::gpt2_transformer::{gpt2_generate_token, GPT2GenerateTokenArgs};
use crate::generate::{GenerateNextTokenArgs, Model};

pub fn load_tensor(model: &mut SafeTensor, name: &str) -> Vecf32 {
    if !model.entries.contains_key(name) {
        panic!("NOT FOUND KEY {}", name);
    }
    let entry = &model.entries[name];
    let start = model.data_offset + entry.offsets[0];
    let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;

    let mut bytes = vec![0u8; byte_len];
    model.file.seek(Start(start)).unwrap();
    model.file.read_exact(&mut bytes).expect("Failed to read file");

    let result = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    result
}

fn load_weight_bias(safe_tensors : &mut SafeTensor, name: &str, layer: usize) -> (Vec<f32>, Vec<f32>) {
    let prefix = format!("h.{}", layer);
    let w = load_tensor(safe_tensors,  &format!("{}.{}.weight", prefix, name));
    let b = load_tensor(safe_tensors,&format!("{}.{}.bias", prefix, name));

    (w, b)
}

pub struct GPT2Layer {
    pub ln1_w: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub proj_w: Weight,
    pub proj_b: Vec<f32>,
    pub ln2_w: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub ffn_up_w: Weight,
    pub ffn_up_b: Vec<f32>,
    pub ffn_down_w: Weight,
    pub ffn_down_b: Vec<f32>,
    pub qkv_w: Weight,
    pub qkv_b: Vec<f32>,
}

impl LoadLayer for GPT2Layer {
    fn load_layer(safe_tensor: &mut SafeTensor, layer: usize, gpu: &Gpu) -> Self {
        let (ln1_w, ln1_b) = load_weight_bias(safe_tensor, "ln_1", layer);
        let (qkv_w, qkv_b) = load_weight_bias(safe_tensor, "attn.c_attn", layer);
        let (proj_w, proj_b) = load_weight_bias(safe_tensor, "attn.c_proj", layer);
        let (ln2_w, ln2_b) = load_weight_bias(safe_tensor, "ln_2", layer);
        let (ffn_up_w, ffn_up_b) = load_weight_bias(safe_tensor, "mlp.c_fc", layer);
        let (ffn_down_w, ffn_down_b) = load_weight_bias(safe_tensor, "mlp.c_proj", layer);

        Self { ln1_w, ln1_b, qkv_w: gpu.upload(&qkv_w), qkv_b, proj_w: gpu.upload(&proj_w), proj_b, ln2_w, ln2_b, ffn_up_w: gpu.upload(&ffn_up_w), ffn_up_b, ffn_down_w: gpu.upload(&ffn_down_w), ffn_down_b }
    }
}

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
    pub fn new(safe_tensors: &mut SafeTensor, gpu: &Gpu) -> Self {
        let config = GPT2Config::load(&format!("{}/config.json", safe_tensors.path_to_model_dir));
        let wte = load_tensor(safe_tensors, "wte.weight");
        let wte_gpu = gpu.upload(&wte);
        let wpe = load_tensor(safe_tensors, "wpe.weight");
        let ln_f_w = load_tensor(safe_tensors, "ln_f.weight");
        let ln_f_b = load_tensor(safe_tensors, "ln_f.bias");
        let layers = load_all_layers(safe_tensors, config.n_layer, &gpu);


        Self {wte, wte_gpu, wpe, ln_f_w, ln_f_b, layers, config}
    }
}

impl Model for GPT2Model {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize {
        let GenerateNextTokenArgs { token_id, position, kv_cache, gpu} = args;
        let res = gpt2_generate_token(GPT2GenerateTokenArgs {token_id, position, kv_cache, gpu, model: self});
        res
    }
}