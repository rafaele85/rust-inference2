use crate::root::arch::qwen35::qwen35_config::Qwen35Config;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::arch::qwen35::qwen35_layer::{Qwen35FullAttnLayer, Qwen35LinearAttnLayer};
use crate::root::arch::safetensors::{load_mlx_affine, load_tensor_to_f32, SafeTensor};
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};

pub enum Qwen35Layer {
    Linear(Qwen35LinearAttnLayer),
    Full(Qwen35FullAttnLayer),
}

pub struct Qwen35Model {
    pub config: Qwen35Config,
    pub embed_weights: Vecf32,
    pub final_norm_w: Vecf32,
    pub lm_head_w: WeightF16,
    pub layers: Vec<Qwen35Layer>,
}

impl Qwen35Model {
    pub fn new(st: &mut SafeTensor, gpus: &mut Vec<Gpu>) -> Self {
        let config = Qwen35Config::load(&format!("{}/config.json", st.path_to_model_dir));
        let bits = config.bits();
        let gs = config.group_size();
        let num_layers = config.text_config.num_hidden_layers;

        println!("Loading embeddings...");
        let embed_weights = load_mlx_affine(st,
                                            "language_model.model.embed_tokens.weight",
                                            "language_model.model.embed_tokens.scales",
                                            "language_model.model.embed_tokens.biases",
                                            bits, gs);

        let final_norm_w = load_tensor_to_f32(st, "language_model.model.norm.weight");

        let default_gpu = get_default_gpu(gpus, "lm_head");
        let lm_head = load_mlx_affine(st,
                                      "language_model.lm_head.weight",
                                      "language_model.lm_head.scales",
                                      "language_model.lm_head.biases",
                                      bits, gs);
        let lm_head_w = default_gpu.upload_f16_from_f32(&lm_head, "lm_head");

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let gpu = get_gpu(gpus, i, num_layers);
            if config.is_linear_attention(i) {
                println!("Loading layer {} (linear)...", i);
                let layer = Qwen35LinearAttnLayer::load(st, i, gpu, &config);
                layers.push(Qwen35Layer::Linear(layer));
            } else {
                println!("Loading layer {} (full)...", i);
                let layer = Qwen35FullAttnLayer::load(st, i, gpu, &config);
                layers.push(Qwen35Layer::Full(layer));
            }
        }

        Self { config, embed_weights, final_norm_w, lm_head_w, layers }
    }
}
