use std::fs::File;
use std::io::Read;

fn main() {
    let mut file = File::open("./model/model.safetensors").expect("Model file not found");
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).expect("Failed to read file");
    let header_len = u64::from_le_bytes(buf);
    println!("Header length {}", header_len);
}
