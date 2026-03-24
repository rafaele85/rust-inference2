use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek};
use std::io::SeekFrom::Start;
use crate::common_types::{Vecf32};

pub struct TensorEntry {
    pub offsets: [u64; 2],
}
pub struct SafeTensor {
    pub file: File,
    pub data_offset: u64,
    pub entries: HashMap<String, TensorEntry>,
    pub path_to_model_dir: String
}
impl SafeTensor {
    pub fn open(path_to_model_dir: &str) -> Self {
        let path_to_safetensors = &format!("{}/model.safetensors", &path_to_model_dir);
        let mut safetensors_file = File::open(path_to_safetensors).expect(&format!("Failed to open file {}", &path_to_safetensors));
        let mut buf = [0u8; 8];
        safetensors_file.read_exact(&mut buf).expect("Failed to read file");
        let header_len = u64::from_le_bytes(buf);
        let mut header_bytes = vec![0u8; header_len as usize];
        safetensors_file.read_exact(&mut header_bytes).expect("Failed to read file");
        let header_str = String::from_utf8(header_bytes).expect("Invalid UTF-8");
        let json: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str).expect("Invalid JSON");

        let mut entries = HashMap::new();
        for (name, value) in &json {
            if name == "__metadata__" {
                continue;
            }
            let offsets: Vec<u64> = serde_json::from_value(value["data_offsets"].clone()).unwrap();
            let tensor_entry = TensorEntry {
                offsets: [offsets[0], offsets[1]],
            };
            entries.insert(name.clone(), tensor_entry);
        }

        let data_offset = 8 + header_len;

        SafeTensor { file: safetensors_file, data_offset, entries, path_to_model_dir: path_to_model_dir.to_string() }
    }


    pub fn load_tensor_bf16(&mut self, name: &str) -> Vecf32 {
        if !self.entries.contains_key(name) {
            panic!("NOT FOUND KEY {}", name);
        }

        let entry = &self.entries[name];
        let start = self.data_offset + entry.offsets[0];
        let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;

        let mut bytes = vec![0u8; byte_len];
        self.file.seek(Start(start)).unwrap();
        self.file.read_exact(&mut bytes).expect("Failed to read file");

        let result = bytes.chunks_exact(2).map(|c| {
            let bits = u16::from_le_bytes(c.try_into().unwrap());
            f32::from_bits((bits as u32) << 16)
        }).collect();

        result
    }
}

