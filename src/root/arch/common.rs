use crate::root::arch::safetensors::SafeTensor;
use crate::root::config::{MLXConfig, ModelDimConfig};
use crate::root::gpu::{get_gpu, Gpu};

pub trait LoadLayer: Sized {
    fn load_layer<TC: MLXConfig>(safe_tensor: &mut SafeTensor, layer_no: usize, config: &TC, gpu: &mut Gpu) -> Self;
}


pub fn load_all_layers<L: LoadLayer, TC: MLXConfig + ModelDimConfig>(model: &mut SafeTensor, config: &TC, gpus: &mut Vec<Gpu>) -> Vec<L> {
    let nlayers = config.get_num_hidden_layers();
    let mut layers = Vec::with_capacity(nlayers);
    for i in 0..nlayers {
        let gpu = get_gpu(gpus, i, nlayers);
        let layer = L::load_layer(model, i, config, gpu);
        layers.push(layer);
    }

    layers
}


