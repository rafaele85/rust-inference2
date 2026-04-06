use std::collections::HashMap;
use std::fs;
use serde::{Deserialize, Deserializer};
use serde::de::Error;
use serde_json::{Map, Value};

enum TokenSegment {
    Special(u32),
    Text(String)
}
pub struct BpeTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
    pub space_char: String,
    pub nl_char: String,
    pub add_leading_space: bool,
    pub added_tokens: Vec<(String, u32)>,
}

impl BpeTokenizer {
    fn split_on_added_tokens(&self, text: &str) -> Vec<TokenSegment> {
        let mut segments = Vec::new();
        let mut remaining = text.to_string();

        while !remaining.is_empty() {
            let mut found = false;
            for (token_str, id) in &self.added_tokens {
                if let Some(pos) = remaining.find(token_str.as_str()) {
                    if pos > 0 {
                        segments.push(TokenSegment::Text(remaining[..pos].to_string()));
                    }
                    segments.push(TokenSegment::Special(*id));
                    remaining = remaining[pos+token_str.len()..].to_string();
                    found = true;
                    break;
                }
            }
            if !found {
                segments.push(TokenSegment::Text(remaining));
                break;
            }
        }
        segments
    }

    fn build_vocab(vocab_map: Map<String, Value>) -> (HashMap<String, u32>, HashMap<u32, String>) {
        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (token, id) in vocab_map {
            let id = id.as_u64().unwrap() as u32;
            vocab.insert(token.clone(), id);
            id_to_token.insert(id, token.clone());
        }

        (vocab, id_to_token)
    }

    fn parse_merge(m: &Value) -> (String, String) {
        if let Some(s) = m.as_str() {
            let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            let parts = m.as_array().unwrap();
            (parts[0].as_str().unwrap().to_string(), parts[1].as_str().unwrap().to_string())
        }
    }

    fn build_added_tokens(json: &Value) -> Vec<(String, u32)> {
        let mut added_tokens: Vec<(String, u32)> = Vec::new();
        if let Some(arr) = json["added_tokens"].as_array() {
            for entry in arr {
                let special = entry["special"].as_bool().unwrap_or(false);
                if special {
                    let content = entry["content"].as_str().unwrap().to_string();
                    let id = entry["id"].as_u64().unwrap() as u32;
                    added_tokens.push((content, id));
                }
            }
        }
        added_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        added_tokens
    }

    pub fn load(base_path: &str, space_char: String, nl_char: String, add_leading_space: bool) -> Self {
        let path = format!("{}/{}", base_path, "tokenizer.json");
        let content = fs::read_to_string(&path).expect(&format!("Failed to read file {}", path));
        let json: Value = serde_json::from_str(&content).expect(&format!("Failed to parse file {}", path));
        let vocab_map = json["model"]["vocab"].as_object().unwrap();

        let (vocab, id_to_token) = Self::build_vocab(vocab_map.clone());
        let added_tokens = BpeTokenizer::build_added_tokens(&json);

        let merges_arr = json["model"]["merges"].as_array().unwrap();

        let merges: Vec<(String, String)> = merges_arr.iter().map(|m| Self::parse_merge(m) ).collect();

        Self {vocab, id_to_token, merges, space_char, nl_char, add_leading_space, added_tokens}
    }

    fn fill_merge_ranks(&self) -> HashMap<(String, String), usize> {
        let mut merge_ranks: HashMap<(String, String), usize> = HashMap::new();
        for (i, (a, b)) in self.merges.iter().enumerate() {
            merge_ranks.insert((a.clone(), b.clone()), i);
        }
        merge_ranks
    }

    fn text_to_chars(&self, text: &str) -> Vec<String> {
        let text = text.trim().replace(" ", &self.space_char);
        let prefix = if self.add_leading_space { &self.space_char } else { &"".to_string() };
        let prepended_text = format!("{}{}", prefix, text);

        let tokens: Vec<String> = prepended_text.bytes()
            .map(|b| byte_to_unicode(b).to_string())
            .collect();

        tokens
    }

    fn find_best_pair_idx(&self, tokens: &mut Vec<String>, merge_ranks: &HashMap<(String, String), usize>) -> Option<usize> {
        let mut best_idx: Option<usize> = None;
        let mut best_rank = usize::MAX;

        for i in 0..tokens.len() - 1 {
            let pair = (tokens[i].clone(), tokens[i + 1].clone());
            if let Some(&rank) = merge_ranks.get(&pair) {
                if rank < best_rank {
                    best_rank = rank;
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

    fn hex_to_id(&self, t: &String) -> Vec<u32> {
        let bytes = t.as_bytes();
        let mut result = Vec::new();

        for &b in bytes {
            let unicode_char = byte_to_unicode(b);
            let unicode_str = unicode_char.to_string();

            if let Some(&id) = self.vocab.get(&unicode_str) {
                result.push(id);
            } else {
                // fallback to <0xHH> format for SentencePiece tokenizers
                let hex = format!("<0x{:02X}>", b);
                let id = *self.vocab.get(&hex).expect(&format!("vocab entry for byte 0x{:02X} not found", b));
                result.push(id);
            }
        }
        result
    }

    fn token_to_ids(&self, t: &String) -> Vec<u32> {
        if let Some(&id) = self.vocab.get(t) {
            vec![id]
        } else {
            self.hex_to_id(t)
        }
    }

    fn tokens_to_ids(&self, tokens: Vec<String>) -> Vec<u32> {
        let res = tokens.iter().flat_map(|t| self.token_to_ids(t)).collect();

        res
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let merge_ranks = self.fill_merge_ranks();
        let segments = self.split_on_added_tokens(text);
        let mut all_ids = Vec::new();
        for segment in segments {
            match segment {
                TokenSegment::Special(id) => all_ids.push(id),
                TokenSegment::Text(t) => {
                    let mut tokens = self.text_to_chars(&t);
                    while self.merge_best(&mut tokens, &merge_ranks) {}
                    let token_ids = self.tokens_to_ids(tokens);
                    all_ids.extend(token_ids);
                }
            }
        }
        all_ids
    }

    fn cleanup_decode(&self, s: &String) -> String {
        s.replace(&self.space_char, " ").replace(&self.nl_char, "\n")
    }
    pub fn decode(&self, ids: &Vec<u32>) -> String {
        let mut token_strings: Vec<String> = Vec::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(&id) {
                token_strings.push(token.clone());
            }
        }

        let text = token_strings.join("");

        let mut bytes: Vec<u8> = Vec::new();
        for c in text.chars() {
            if let Some(b) = unicode_to_byte(c) {
                bytes.push(b);
            } else {
                // multi-byte char not in mapping, push its UTF-8 bytes directly
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                bytes.extend_from_slice(s.as_bytes());
            }
        }

        let text = String::from_utf8_lossy(&bytes).to_string();
        let res = self.cleanup_decode(&text);
        
        res
    }
}



#[derive(Deserialize)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>)
}

pub fn deserialize_eos_token_id<'de, D>(deserializer: D) -> Result<Vec<u32>, D::Error>
where D: Deserializer<'de>,
{
    match EosTokenId::deserialize(deserializer)? {
        EosTokenId::Single(n) => Ok(vec![n]),
        EosTokenId::Multiple(arr) => Ok(arr),
    }
}

const BYTES_TO_UNICODE: [char; 256] = [
    '\u{0100}', '\u{0101}', '\u{0102}', '\u{0103}', '\u{0104}', '\u{0105}', '\u{0106}', '\u{0107}', '\u{0108}', '\u{0109}', '\u{010A}', '\u{010B}', '\u{010C}', '\u{010D}', '\u{010E}', '\u{010F}',
    '\u{0110}', '\u{0111}', '\u{0112}', '\u{0113}', '\u{0114}', '\u{0115}', '\u{0116}', '\u{0117}', '\u{0118}', '\u{0119}', '\u{011A}', '\u{011B}', '\u{011C}', '\u{011D}', '\u{011E}', '\u{011F}',
    '\u{0120}', '\u{0021}', '\u{0022}', '\u{0023}', '\u{0024}', '\u{0025}', '\u{0026}', '\u{0027}', '\u{0028}', '\u{0029}', '\u{002A}', '\u{002B}', '\u{002C}', '\u{002D}', '\u{002E}', '\u{002F}',
    '\u{0030}', '\u{0031}', '\u{0032}', '\u{0033}', '\u{0034}', '\u{0035}', '\u{0036}', '\u{0037}', '\u{0038}', '\u{0039}', '\u{003A}', '\u{003B}', '\u{003C}', '\u{003D}', '\u{003E}', '\u{003F}',
    '\u{0040}', '\u{0041}', '\u{0042}', '\u{0043}', '\u{0044}', '\u{0045}', '\u{0046}', '\u{0047}', '\u{0048}', '\u{0049}', '\u{004A}', '\u{004B}', '\u{004C}', '\u{004D}', '\u{004E}', '\u{004F}',
    '\u{0050}', '\u{0051}', '\u{0052}', '\u{0053}', '\u{0054}', '\u{0055}', '\u{0056}', '\u{0057}', '\u{0058}', '\u{0059}', '\u{005A}', '\u{005B}', '\u{005C}', '\u{005D}', '\u{005E}', '\u{005F}',
    '\u{0060}', '\u{0061}', '\u{0062}', '\u{0063}', '\u{0064}', '\u{0065}', '\u{0066}', '\u{0067}', '\u{0068}', '\u{0069}', '\u{006A}', '\u{006B}', '\u{006C}', '\u{006D}', '\u{006E}', '\u{006F}',
    '\u{0070}', '\u{0071}', '\u{0072}', '\u{0073}', '\u{0074}', '\u{0075}', '\u{0076}', '\u{0077}', '\u{0078}', '\u{0079}', '\u{007A}', '\u{007B}', '\u{007C}', '\u{007D}', '\u{007E}', '\u{0121}',
    '\u{0122}', '\u{0123}', '\u{0124}', '\u{0125}', '\u{0126}', '\u{0127}', '\u{0128}', '\u{0129}', '\u{012A}', '\u{012B}', '\u{012C}', '\u{012D}', '\u{012E}', '\u{012F}', '\u{0130}', '\u{0131}',
    '\u{0132}', '\u{0133}', '\u{0134}', '\u{0135}', '\u{0136}', '\u{0137}', '\u{0138}', '\u{0139}', '\u{013A}', '\u{013B}', '\u{013C}', '\u{013D}', '\u{013E}', '\u{013F}', '\u{0140}', '\u{0141}',
    '\u{0142}', '\u{00A1}', '\u{00A2}', '\u{00A3}', '\u{00A4}', '\u{00A5}', '\u{00A6}', '\u{00A7}', '\u{00A8}', '\u{00A9}', '\u{00AA}', '\u{00AB}', '\u{00AC}', '\u{0143}', '\u{00AE}', '\u{00AF}',
    '\u{00B0}', '\u{00B1}', '\u{00B2}', '\u{00B3}', '\u{00B4}', '\u{00B5}', '\u{00B6}', '\u{00B7}', '\u{00B8}', '\u{00B9}', '\u{00BA}', '\u{00BB}', '\u{00BC}', '\u{00BD}', '\u{00BE}', '\u{00BF}',
    '\u{00C0}', '\u{00C1}', '\u{00C2}', '\u{00C3}', '\u{00C4}', '\u{00C5}', '\u{00C6}', '\u{00C7}', '\u{00C8}', '\u{00C9}', '\u{00CA}', '\u{00CB}', '\u{00CC}', '\u{00CD}', '\u{00CE}', '\u{00CF}',
    '\u{00D0}', '\u{00D1}', '\u{00D2}', '\u{00D3}', '\u{00D4}', '\u{00D5}', '\u{00D6}', '\u{00D7}', '\u{00D8}', '\u{00D9}', '\u{00DA}', '\u{00DB}', '\u{00DC}', '\u{00DD}', '\u{00DE}', '\u{00DF}',
    '\u{00E0}', '\u{00E1}', '\u{00E2}', '\u{00E3}', '\u{00E4}', '\u{00E5}', '\u{00E6}', '\u{00E7}', '\u{00E8}', '\u{00E9}', '\u{00EA}', '\u{00EB}', '\u{00EC}', '\u{00ED}', '\u{00EE}', '\u{00EF}',
    '\u{00F0}', '\u{00F1}', '\u{00F2}', '\u{00F3}', '\u{00F4}', '\u{00F5}', '\u{00F6}', '\u{00F7}', '\u{00F8}', '\u{00F9}', '\u{00FA}', '\u{00FB}', '\u{00FC}', '\u{00FD}', '\u{00FE}', '\u{00FF}',
];


fn byte_to_unicode(b: u8) -> char {
    BYTES_TO_UNICODE[b as usize]
}


fn unicode_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    if code >= 0x0100 && code <= 0x0120 { return Some((code - 0x0100) as u8); }
    if code >= 0x0021 && code <= 0x007E { return Some((code - 0x0021 + 33) as u8); }
    if code >= 0x0121 && code <= 0x0142 { return Some((code - 0x0121 + 127) as u8); }
    if code >= 0x00A1 && code <= 0x00AC { return Some((code - 0x00A1 + 161) as u8); }
    if code == 0x0143 { return Some(173); }
    if code >= 0x00AE && code <= 0x00FF { return Some((code - 0x00AE + 174) as u8); }
    None
}
