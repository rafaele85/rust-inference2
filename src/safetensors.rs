use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::io::SeekFrom::Start;

struct TensorEntry {
    offsets: [u64; 2],
    shape: Vec<usize>
}
pub struct SafeTensor {
    file: File,
    data_offset: u64,
    entries: HashMap<String, TensorEntry>
}

impl SafeTensor {
    pub fn open(path: &str) -> Self {
        let mut file = File::open(path).expect("Failed to open file");
        let mut buf = [0u8; 8];
        file.read_exact(&mut buf).expect("Failed to read file");
        let header_len = u64::from_le_bytes(buf);
        let mut header_bytes = vec![0u8; header_len as usize];
        file.read_exact(&mut header_bytes).expect("Failed to read file");
        let header_str = String::from_utf8(header_bytes).expect("Invalid UTF-8");
        let json: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str).expect("Invalid JSON");

        let mut entries = HashMap::new();
        for (name, value) in &json {
            if name == "__metadata__" {
                continue;
            }
            let offsets: Vec<u64> = serde_json::from_value(value["data_offsets"].clone()).unwrap();
            let shape: Vec<usize> = serde_json::from_value(value["shape"].clone()).unwrap();
            let tensor_entry = TensorEntry {
                offsets: [offsets[0], offsets[1]],
                shape
            };
            entries.insert(name.clone(), tensor_entry);
        }

        let data_offset = 8 + header_len as u64;
        SafeTensor { file, data_offset, entries }
    }

    pub fn load_tensor(&mut self, name: &str) -> Vec<f32> {
        if !self.entries.contains_key(name) {
            println!("NOT FOUND KEY {}", name);
        }
        let entry = &self.entries[name];
        let start = self.data_offset + entry.offsets[0];
        let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;

        let mut bytes = vec![0u8; byte_len];
        self.file.seek(Start(start)).unwrap();
        self.file.read_exact(&mut bytes).expect("Failed to read file");

        let result = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

        result
    }
}


fn load_weight_bias(model: &mut SafeTensor, name: &str, layer: usize) -> (Vec<f32>, Vec<f32>) {
    let prefix = format!("h.{}", layer);
    let w = model.load_tensor( &format!("{}.{}.weight", prefix, name));
    let b = model.load_tensor(&format!("{}.{}.bias", prefix, name));

    (w, b)
}
pub fn load_layer(model: &mut SafeTensor, layer: usize) -> (
  Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,Vec<f32>, Vec<f32>,
) {
    let (ln1_w, ln1_b) = load_weight_bias(model, "ln_1", layer);
    let (qkv_w, qkv_b) = load_weight_bias(model, "attn.c_attn", layer);
    let (proj_w, proj_b) = load_weight_bias(model, "attn.c_proj", layer);
    let (ln2_w, ln2_b) = load_weight_bias(model, "ln_2", layer);
    let (ffn_up_w, ffn_up_b) = load_weight_bias(model, "mlp.c_fc", layer);
    let (ffn_down_w, ffn_down_b) = load_weight_bias(model, "mlp.c_proj", layer);

    (ln1_w, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_w, ln2_b, ffn_up_w, ffn_up_b, ffn_down_w, ffn_down_b)
}