use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

fn main() {
    let mut file = File::open("./model/model.safetensors").expect("Model file not found");
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).expect("Failed to read file");
    let header_len = u64::from_le_bytes(buf) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).expect("Failed to read file");
    let header_str = String::from_utf8(header_bytes).expect("Failed to parse header bytes");

    let json: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str).expect("Failed to parse header");

    let data_offset = 8 + header_len as u64;
    let tensor_name = "wte.weight"; //"ln_f.bias";
    let entry = &json[tensor_name];
    let offsets_orig = entry["data_offsets"].clone();

    //println!("raw entry: {}", entry);

    let offsets: Vec<u64> = serde_json::from_value(offsets_orig).expect("Failed to parse offsets");

    //println!("offsets: {:?}", offsets);

    let start = data_offset + offsets[0];
    let end = data_offset + offsets[1];
    let byte_len = (end - start) as usize;

    //println!("byte_len: {}", byte_len);

    let mut names: Vec<&String> = json.keys().collect();
    names.sort();
    for name in &names {
        if *name == "__metadata__" {
            continue;
        }
        let entry = &json[*name];
        let shape_orig = entry["shape"].clone();
        let shape: Vec<i64> = serde_json::from_value(shape_orig).expect("Failed to parse shape");

        let offsets_orig = entry["data_offsets"].clone();
        let offsets: Vec<i64> = serde_json::from_value(offsets_orig).expect("Failed to parse offsets");
        //println!("{} {:?} [{}, {}]", name, shape, offsets[0], offsets[1]);
    }

    let mut tensor_bytes = vec![0u8; byte_len];
    file.seek(SeekFrom::Start(start)).expect("Failed to seek");
    file.read_exact(&mut tensor_bytes).expect("Failed to read file");

    let floats: Vec<f32> = tensor_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();

    let token_id = 464;
    let dim = 768;
    let row_start = token_id * dim;

    for v in &floats[row_start..row_start + 10] {
        println!("{:.6}", v)
    }



}
