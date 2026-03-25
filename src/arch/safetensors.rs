use std::collections::HashMap;
use std::fs::File;
use std::io::{Read};

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
    fn read_header(safetensor_file: &mut File, ) -> (u64, HashMap<String, serde_json::value::Value>)  {
        let mut buf = [0u8; 8];
        safetensor_file.read_exact(&mut buf).expect("Failed to read file");
        let header_len = u64::from_le_bytes(buf);
        let mut header_bytes = vec![0u8; header_len as usize];
        safetensor_file.read_exact(&mut header_bytes).expect("Failed to read file");
        let header_str = String::from_utf8(header_bytes).expect("Invalid UTF-8");
        let json: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str).expect("Invalid JSON");

        (header_len, json)
    }

    fn prepare_entries(header: HashMap<String, serde_json::value::Value>) -> HashMap<String, TensorEntry> {
        let mut entries = HashMap::new();
        for (name, value) in &header {
            if name == "__metadata__" {
                continue;
            }
            let offsets: Vec<u64> = serde_json::from_value(value["data_offsets"].clone()).unwrap();
            let tensor_entry = TensorEntry {
                offsets: [offsets[0], offsets[1]],
            };
            entries.insert(name.clone(), tensor_entry);
        }

        entries
    }

    pub fn open(path_to_model_dir: &str) -> Self {
        let path_to_safetensors = &format!("{}/model.safetensors", &path_to_model_dir);
        let mut safetensors_file = File::open(path_to_safetensors).expect(&format!("Failed to open file {}", &path_to_safetensors));

        let (header_len, header) = SafeTensor::read_header(&mut safetensors_file);
        let entries = Self::prepare_entries(header);

        let data_offset = 8 + header_len;

        SafeTensor { file: safetensors_file, data_offset, entries, path_to_model_dir: path_to_model_dir.to_string() }
    }
}

