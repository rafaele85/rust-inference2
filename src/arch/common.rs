use crate::gpu::Gpu;
use crate::arch::safetensors::SafeTensor;

pub trait LoadLayer: Sized {
    fn load_layer(safe_tensor: &mut SafeTensor, layer_no: usize, gpu: &Gpu) -> Self;
}


pub fn load_all_layers<L: LoadLayer>(model: &mut SafeTensor, nlayers: usize, gpu: &Gpu) -> Vec<L> {

    let mut layers = Vec::with_capacity(nlayers);
    for i in 0..nlayers {
        let layer = L::load_layer(model, i, gpu);
        layers.push(layer);
    }

    layers
}


