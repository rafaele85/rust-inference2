use std::collections::HashMap;
use std::env;
use std::fs::{read_to_string, File};
use std::io::{Read};
use std::path::Path;
use half::f16;
use memmap2::Mmap;
use serde_json::{Map, Value};
use crate::root::arch::quant::awq::{awq_dequantize, AwqDequantizeArgs, AWQ_PACK_SIZE};
use crate::root::arch::quant::compressed_tensors::ct_dequantize;
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::gpu::Gpu;

pub struct TensorEntry {
    pub offsets: [u64; 2],
    pub shard_idx: usize,
    pub shape: Vec<usize>,
    pub dtype: String,
}
pub struct SafeTensor {
    pub mmaps: Vec<Mmap>,
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
            let shape: Vec<usize> = serde_json::from_value(value["shape"].clone()).unwrap();
            let dtype: String = value["dtype"].as_str().unwrap().to_string();
            let tensor_entry = TensorEntry {
                offsets: [offsets[0], offsets[1]],
                shard_idx,
                shape,
                dtype
            };
            entries.insert(name.clone(), tensor_entry);
        }

        entries
    }

    pub fn open(path_to_model_dir: &str) -> Self {
        let path_to_safetensors = &format!("{}/model.safetensors", &path_to_model_dir);
        let mut file = File::open(path_to_safetensors).expect(&format!("Failed to open file {}", &path_to_safetensors));
        let mmap = unsafe { Mmap::map(&file) }.expect(&format!("Failed to mmap file {}", &path_to_safetensors));

        let (header_len, header) = SafeTensor::read_header(&mut file);
        let entries = Self::prepare_entries(header, 0);

        let data_offset = 8 + header_len;

        SafeTensor { mmaps: vec![mmap], data_offsets: vec![data_offset], entries, path_to_model_dir: path_to_model_dir.to_string() }
    }

    fn prepare_shard_paths(path_to_model_dir: &str, weight_map: &Map<String, Value>) -> Vec<String> {
        let mut paths: Vec<String> = weight_map.values().map(|v|
            format!("{}/{}", path_to_model_dir, v.as_str().unwrap().to_string())
        ).collect();
        paths.sort();
        paths.dedup();

        paths
    }


    pub fn open_sharded(path_to_model_dir: &str) -> Self {
        let index_path = &format!("{}/model.safetensors.index.json", path_to_model_dir);
        let index_str = read_to_string(index_path).expect(&format!("Failed to open file {}", &index_path));
        let index_json: Value = serde_json::from_str(&index_str).expect("Invalid JSON");
        let weight_map = index_json["weight_map"].as_object().unwrap();
        let mut entries = HashMap::new();
        let mut data_offsets = Vec::new();
        let mut mmaps = Vec::new();

        let shard_paths = SafeTensor::prepare_shard_paths(path_to_model_dir, weight_map);

        for (shard_idx, shard_path) in shard_paths.iter().enumerate() {
            let mut file = File::open(shard_path).expect(&format!("Failed to open file {}", &shard_path));
            let mmap = unsafe { Mmap::map(&file) }.expect(&format!("Failed to mmap file {}", shard_path));
            let (header_len, header) = Self::read_header(&mut file);
            let shard_entries = SafeTensor::prepare_entries(header, shard_idx);
            entries.extend(shard_entries);
            data_offsets.push(8 + header_len);
            mmaps.push(mmap);
        }

        SafeTensor { mmaps, data_offsets, entries, path_to_model_dir: path_to_model_dir.to_string() }
    }

    pub fn new(path_to_model_dir: &str) -> Self {
        let single = &format!("{}/model.safetensors", path_to_model_dir);
        if Path::new(single).exists() {
            Self::open(path_to_model_dir)
        } else {
            Self::open_sharded(path_to_model_dir)
        }
    }

    pub fn load_layer_weight(&mut self, name: &str, layer: usize) -> Vecf32 {
        let prefix = format!("model.layers.{}", layer);
        let bf16_key = format!("{}.{}.weight", prefix, name);
        let packed_key = format!("{}.{}.weight_packed", prefix, name);
        let scale_key = format!("{}.{}.weight_scale", prefix, name);
        let shape_key = format!("{}.{}.weight_shape", prefix, name);
        let awq_key = format!("{}.{}.qweight", prefix, name);

        if self.entries.contains_key(&bf16_key) {
            // Layer 0: regular bfloat16
            load_tensor_to_f32(self, &bf16_key)
        } else if self.entries.contains_key(&packed_key) {
            // Layers 1+: AWQ packed format
            let packed = load_tensor_i32(self, &packed_key);
            let scales = load_tensor_f32(self, &scale_key); // or f16 then convert
            let shape = load_tensor_i32(self, &shape_key);  // [out_dim, in_dim]

            let out_dim = shape[0] as usize;
            let in_dim = shape[1] as usize;

            ct_dequantize(&packed, &scales, out_dim, in_dim)
        } else if self.entries.contains_key(&awq_key) {
            load_tensor_awq(self, &format!("{}.{}", prefix, name))
        } else {
            panic!("Not found weight for {}.{}", prefix, name)
        }
    }


    pub fn load_dense_ffns(&mut self, layer_no: usize, gpu: &mut Gpu) -> (Option<WeightF16>, Option<WeightF16>, Option<WeightF16>) {
        let has_dense_ffns = self.entries.contains_key(
            &format!("model.layers.{}.mlp.gate_proj.weight", layer_no)
        ) || self.entries.contains_key(
            &format!("model.layers.{}.mlp.gate_proj.qweight", layer_no)
        );
        if has_dense_ffns  {
            let gate_proj_w = gpu.upload_f16_from_f32( &self.load_layer_weight("mlp.gate_proj", layer_no), "gate_proj_w" );
            let up_proj_w = gpu.upload_f16_from_f32( &self.load_layer_weight( "mlp.up_proj", layer_no), "up_proj_w");
            let down_proj_w = gpu.upload_f16_from_f32( &self.load_layer_weight( "mlp.down_proj", layer_no), "down_proj_w" );

            ( Some(gate_proj_w), Some(up_proj_w), Some(down_proj_w) )
        } else {
            (None, None, None)
        }
    }


    pub fn load_weight_optional(&mut self, name: &str, layer: usize) -> Option<Vecf32> {
        let entries = &self.entries;
        let key = format!("model.layers.{}.{}.weight", layer, name);
        if entries.contains_key(&key) {
            let b = load_tensor_to_f32(self, &key);
            return Some(b);
        }
        None
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



pub fn load_tensor_raw<'a>(safe_tensor: &'a mut SafeTensor, name: &str) -> &'a [u8] {
    if !safe_tensor.entries.contains_key(name) {
        panic!("NOT FOUND KEY {}", name);
    }

    let entry = &safe_tensor.entries[name];
    let shard_idx = entry.shard_idx;
    let start = safe_tensor.data_offsets[shard_idx] as usize + entry.offsets[0] as usize;
    let byte_len = (entry.offsets[1] - entry.offsets[0]) as usize;

    let bytes = &safe_tensor.mmaps[shard_idx][start..start+byte_len];

    bytes
}


pub fn load_tensor_f32(model: &mut SafeTensor, name: &str) -> Vecf32 {
    let bytes = load_tensor_raw(model, name);

    let result = bytes.chunks_exact(4).map(|c|
        f32::from_le_bytes(c.try_into().unwrap())
    ).collect();

    result
}


pub fn load_tensor_bf16(safe_tensor: &mut SafeTensor, name: &str) -> Vecf32 {
    let bytes = load_tensor_raw(safe_tensor, name);

    let result = bytes.chunks_exact(2).map(|c| {
        let bits = u16::from_le_bytes(c.try_into().unwrap());
        f32::from_bits((bits as u32) << 16)
    }).collect();

    result
}

pub fn load_tensor_i32(safe_tensor: &mut SafeTensor, name: &str) -> Vec<i32> {
    let bytes = load_tensor_raw(safe_tensor, name);
    let result = bytes.chunks_exact(4).map(|c|
      i32::from_le_bytes(c.try_into().unwrap())
    ).collect();

    result
}


pub fn load_tensor_f16(safe_tensor: &mut SafeTensor, name: &str) -> Vecf32 {
    let bytes = load_tensor_raw(safe_tensor, name);

    let result = bytes.chunks_exact(2).map(|c| {
        let bits = u16::from_le_bytes(c.try_into().unwrap());
        f16::from_bits(bits).to_f32()
    }).collect();

    result
}

pub fn load_tensor_awq(safe_tensor: &mut SafeTensor, name: &str) -> Vecf32 {
    let qw_key = format!("{}.qweight", name);
    let qw = load_tensor_i32(safe_tensor, &qw_key);

    let qz_key = format!("{}.qzeros", name);
    let qz = load_tensor_i32(safe_tensor, &qz_key);

    let sc_key = format!("{}.scales", name);
    let sc = load_tensor_f16(safe_tensor, &sc_key);

    let qw_shape = &safe_tensor.entries[&qw_key].shape;
    let in_dim = qw_shape[0];
    let out_dim = qw_shape[1] * AWQ_PACK_SIZE;

    let sc_shape = &safe_tensor.entries[&sc_key].shape;
    let num_group = sc_shape[0];
    let group_size = in_dim / num_group;

    let w = awq_dequantize(AwqDequantizeArgs {
        qweight: &qw, qzero: &qz, scales: &sc, in_dim, out_dim, group_size
    });
    w
}

pub fn load_tensor_to_f32(safe_tensor: &mut SafeTensor, name: &str) -> Vecf32 {
    let dtype = safe_tensor.entries[name].dtype.clone();
    match dtype.as_str() {
        "BF16" => load_tensor_bf16(safe_tensor, name),
        "F16" => load_tensor_f16(safe_tensor, name),
        "F32" => load_tensor_f32(safe_tensor, name),
        _ => panic!("Unknown type {} in tensor {} ", dtype, name)
    }
}


