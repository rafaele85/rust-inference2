use std::collections::HashMap;
use std::fs;

pub struct BpeTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
    pub space_char: String,
    pub add_leading_space: bool
}

impl BpeTokenizer {
    pub fn load(path: &str, space_char: String, add_leading_space: bool) -> Self {
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

        Self {vocab, id_to_token, merges, space_char, add_leading_space}
    }

    fn fill_merge_ranks(&self) -> HashMap<(String, String), usize> {
        let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
        for (i, (a, b)) in self.merges.iter().enumerate() {
            merge_ranks.insert((a.clone(), b.clone()), i as usize);
        }
        merge_ranks
    }

    fn text_to_chars(&self, text: &str) -> Vec<String> {
        let text = text.trim().replace(" ", &self.space_char);
        let prefix = if self.add_leading_space { &self.space_char } else { &"".to_string() };
        let prepended_text = format!("{}{}", prefix, text);
        let mut tokens : Vec<String> = prepended_text.chars().map(|c| c.to_string()).collect();

        tokens
    }

    fn find_best_pair_idx(&self, tokens: &mut Vec<String>, merge_ranks: &HashMap<(String, String), usize>) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_rank = usize::MAX;
        let mut found = false;

        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            if let Some(&rank) = merge_ranks.get(&pair) {
                if rank < best_rank {
                    best_rank = rank;
                    found = true;
                    best_idx = Some(i);
                }
            }
        }

        best_idx
    }
    fn merge_best(&self, tokens: &mut Vec<String>, merge_ranks: &HashMap<(String, String), usize>) -> bool {
        let res = self.find_best_pair_idx(tokens, merge_ranks);

        if let Some(best_idx) = res {
            let merged = format!("{}{}", tokens[best_idx], tokens[best_idx+1]);
            tokens[best_idx] = merged;
            tokens.remove(best_idx + 1);

            return true;
        }

        false
    }

    fn tokens_to_ids(&self, tokens: Vec<String>) -> Vec<u32> {
        let res = tokens.iter().map(|t| {
            *self.vocab.get(t).expect(&format!("vocab entry for {} not found", t))
        }).collect();

        res
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let merge_ranks = self.fill_merge_ranks();
        let mut tokens = self.text_to_chars(text);

        while self.merge_best(&mut tokens, &merge_ranks) {}

        let res = self.tokens_to_ids(tokens);

        res
    }

    pub fn decode(&self, ids: &Vec<u32>) -> String {
        ids.iter().map(|&id| self.id_to_token[&id].replace(&self.space_char, " ")).collect::<Vec<_>>().join("").trim_start().to_string()
    }
}