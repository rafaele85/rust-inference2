use std::collections::HashMap;
use std::env;
use std::fs::{read_to_string, File};
use std::io::{Read};
use std::path::Path;
use serde_json::{Map, Value};

pub struct TensorEntry {
    pub offsets: [u64; 2],
    pub shard_idx: usize,
}
pub struct SafeTensor {
    pub files: Vec<File>,
    pub data_offsets: Vec<u64>,
    pub entries: HashMap<String, TensorEntry>,
    pub path_to_model_dir: String
}
impl SafeTensor {
    fn read_header(safetensor_file: &mut File) -> (u64, HashMap<String, Value>)  {
        let mut buf = [0u8; 8];
        safetensor_file.read_exact(&mut buf).expect("Failed to read file");
        let header_len = u64::from_le_bytes(buf);
        let mut header_bytes = vec![0u8; header_len as usize];
        safetensor_file.read_exact(&mut header_bytes).expect("Failed to read file");
        let header_str = String::from_utf8(header_bytes).expect("Invalid UTF-8");
        let json: HashMap<String, Value> = serde_json::from_str(&header_str).expect("Invalid JSON");

        (header_len, json)
    }

    fn prepare_entries(header: HashMap<String, Value>, shard_idx: usize) -> HashMap<String, TensorEntry> {
        let mut entries = HashMap::new();
        for (name, value) in &header {
            if name == "__metadata__" {
                continue;
            }
            let offsets: Vec<u64> = serde_json::from_value(value["data_offsets"].clone()).unwrap();
            let tensor_entry = TensorEntry {
                offsets: [offsets[0], offsets[1]],
                shard_idx
            };
            entries.insert(name.clone(), tensor_entry);
        }

        entries
    }

    pub fn open(path_to_model_dir: &str) -> Self {
        let path_to_safetensors = &format!("{}/model.safetensors", &path_to_model_dir);
        let mut safetensors_file = File::open(path_to_safetensors).expect(&format!("Failed to open file {}", &path_to_safetensors));

        let (header_len, header) = SafeTensor::read_header(&mut safetensors_file);
        let entries = Self::prepare_entries(header, 0);

        let data_offset = 8 + header_len;

        SafeTensor { files: vec![safetensors_file], data_offsets: vec![data_offset], entries, path_to_model_dir: path_to_model_dir.to_string() }
    }

    fn extract_shard_names(weight_map: &Map<String, Value>) -> Vec<String> {
        let mut names: Vec<String> = weight_map.values().map(|v| v.as_str().unwrap().to_string()).collect();
        names.sort();
        names.dedup();

        names
    }

    pub fn open_sharded(path_to_model_dir: &str) -> Self {
        let index_path = &format!("{}/model.safetensors.index.json", path_to_model_dir);
        let index_str = read_to_string(index_path).expect(&format!("Failed to open file {}", &index_path));
        let index_json: Value = serde_json::from_str(&index_str).expect("Invalid JSON");
        let weight_map = index_json["weight_map"].as_object().unwrap();
        let mut entries = HashMap::new();
        let mut data_offsets = Vec::new();
        let mut files = Vec::new();

        let shard_names = SafeTensor::extract_shard_names(weight_map);
        for (shard_idx, shard_name) in shard_names.iter().enumerate() {
            let shard_path = &format!("{}/{}", path_to_model_dir, shard_name);
            let mut file = File::open(shard_path).expect(&format!("Failed to open file {}", &shard_path));
            let (header_len, header) = Self::read_header(&mut file);
            let shard_entries = SafeTensor::prepare_entries(header, shard_idx);
            entries.extend(shard_entries);
            data_offsets.push(8 + header_len);
            files.push(file);
        }

        SafeTensor { files, data_offsets, entries, path_to_model_dir: path_to_model_dir.to_string() }
    }

    pub fn new(path_to_model_dir: &str) -> Self {
        let single = &format!("{}/model.safetensors", path_to_model_dir);
        if Path::new(single).exists() {
            Self::open(path_to_model_dir)
        } else {
            Self::open_sharded(path_to_model_dir)
        }
    }
}

pub fn resolve_hf_path(model_name: &str) -> String {
    let home = env::var("HOME").expect("HOME env var missing");
    let sanitized = model_name.replace("/", "--");
    let model_dir = format!("{}/.cache/huggingface/hub/models--{}", home, sanitized);
    let ref_path = format!("{}/refs/main", model_dir);
    let hash = read_to_string(&ref_path).expect(&format!("Failed to read file {}", &ref_path))
        .trim().to_string();
    let res = format!("{}/snapshots/{}", model_dir, hash);

    res
}