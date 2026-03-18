use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("./model/model.safetensors").expect("Model file not found");
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).expect("Failed to read file");
    let header_len = u64::from_le_bytes(buf) as usize;

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).expect("Failed to read file");
    let header_str = String::from_utf8(header_bytes).expect("Failed to parse header bytes");

    let json: HashMap<String, serde_json::Value> = serde_json::from_str(&header_str).expect("Failed to parse header");
    let mut names: Vec<&String> = json.keys().collect();
    names.sort();
    for name in &names {
        if *name != "__metadata__" {
            println!("{}", name);
        }
    }
}
