use std::collections::HashMap;
use std::fs;

pub struct Gpt2Vocab {
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
}

impl Gpt2Vocab {
    pub fn load(path: &String) -> Self {

        let content = fs::read_to_string(path).expect(&format!("Failed to read file {}", &path ));
        let token_to_id: HashMap<String, usize> = serde_json::from_str(&content).expect("Failed to parse file");

        let mut id_to_token = HashMap::new();
        for (token, id) in &token_to_id {
            id_to_token.insert(*id, token.clone());
        }

        Gpt2Vocab { id_to_token, token_to_id }
    }

    pub fn decode(&self, token_id: usize) -> String {
        let token = &self.id_to_token[&token_id];
        token.replace("Ġ", " ")
    }

    pub fn encode(&self, str: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        let mut remaining = str;

        while !remaining.is_empty() {
            let mut found = false;
            let max_len = remaining.len();
            let range = (1..=max_len).rev();
            for len in range {
                let candidate = remaining[..len].replace(" ", "Ġ");
                let lookup = self.token_to_id.get(&candidate);
                if let Some(id) = lookup {
                    ids.push(*id);
                    remaining = &remaining[len..];
                    found = true;
                    break;
                }
            }
            if !found {
                panic!("Can not tokenize {}", str);
            }
        }

        ids
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}