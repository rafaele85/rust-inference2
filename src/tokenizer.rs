use std::collections::HashMap;
use std::fs;

pub struct BpeTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(&path).expect(&format!("Failed to read file {}", path));
        let json:serde_json::Value = serde_json::from_str(&content).expect(&format!("Failed to parse file {}", path));
        let vocab_map = json["model"]["vocab"].as_object().unwrap();
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (token, id) in vocab_map {
            let id = id.as_u64().unwrap() as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        let merges_arr = json["model"]["merges"].as_array().unwrap();
        let merges: Vec<(String, String)> = merges_arr.iter().map(|m| {
            let s = m.as_str().unwrap();
            let parts: Vec<&str> = s.splitn(2, ' ').collect();
            (parts[0].to_string(), parts[1].to_string())
        }).collect();

        Self {vocab, id_to_token, merges}
    }
}