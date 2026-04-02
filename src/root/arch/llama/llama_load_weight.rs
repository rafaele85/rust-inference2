use crate::root::arch::safetensors::{load_tensor_awq, load_tensor_to_f32, SafeTensor};
use crate::root::common_types::Vecf32;

pub fn llama_load_bias(safe_tensor : &mut SafeTensor, name: &str, layer: usize) -> Option<Vecf32> {
    let entries = &safe_tensor.entries;
    let key = format!("model.layers.{}.{}.bias", layer, name);
    if entries.contains_key(&key) {
        let b = load_tensor_to_f32(safe_tensor, &key);
        return Some(b);
    }
    None
}

