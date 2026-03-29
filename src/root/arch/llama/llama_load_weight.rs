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

pub fn llama_load_weight_optional(safe_tensor : &mut SafeTensor, name: &str, layer: usize) -> Option<Vecf32> {
    let entries = &safe_tensor.entries;
    let key = format!("model.layers.{}.{}.weight", layer, name);
    if entries.contains_key(&key) {
        let b = load_tensor_to_f32(safe_tensor, &key);
        return Some(b);
    }
    None
}

pub fn llama_load_weight_f16(safe_tensor: &mut SafeTensor, name: &str, layer: usize) -> Vecf32 {
    let prefix = format!("model.layers.{}", layer);
    let w = load_tensor_to_f32(safe_tensor, &format!("{}.{}.weight", prefix, name));
    w
}
