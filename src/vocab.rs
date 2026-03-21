use std::collections::HashMap;
use std::fs;

pub struct Vocab {
    id_to_token: HashMap<usize, String>,
}

impl Vocab {
    pub fn load(path: &str) -> Self {

        let content = fs::read_to_string(path).expect("Failed to read file");
        let mut token_to_id: HashMap<String, usize> = serde_json::from_str(&content).expect("Failed to parse file");

        let mut id_to_token = HashMap::new();
        for (token, id) in token_to_id {
            id_to_token.insert(id, token);
        }

        Vocab { id_to_token }
    }

    pub fn decode(&self, token_id: usize) -> &str {
        &self.id_to_token[&token_id]
    }
}