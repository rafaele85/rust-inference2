use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::tokenizer::BpeTokenizer;

pub struct GenerateNextTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>
}
pub trait Model {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize;
}

fn initial_fill(input_ids: &Vec<u32>, model: &impl Model, kv_cache: &mut KVCache, gpus: &mut Vec<Gpu>) -> u32 {
    let mut last_generated = 0;

    for i in 0..input_ids.len() {
        let token_id = input_ids[i] as usize;
        last_generated = model.generate_next_token(GenerateNextTokenArgs {token_id, position: i, kv_cache, gpus});
    }

    last_generated as u32
}

fn generate_single_pass(all_ids: &mut Vec<u32>, model: &impl Model, eos_token_id: &Vec<u32>, kv_cache: &mut KVCache, gpus: &mut Vec<Gpu>) -> bool {
    let last_id = *all_ids.last().unwrap();
    if eos_token_id.contains(&last_id)  {
        return true;
    }
    let pos = all_ids.len() - 1;
    let top_id = model.generate_next_token(GenerateNextTokenArgs {token_id: last_id as usize, position: pos, kv_cache, gpus});
    all_ids.push(top_id as u32);

    false
}

pub struct GenerateArgs<'a> {
    pub prompt: &'a str,
    pub tokenizer: &'a BpeTokenizer,
    pub max_tokens: usize,
    pub gpus: &'a mut Vec<Gpu>,
    pub eos_token_id: &'a Vec<u32>,
    pub num_layers: usize,
}

pub fn generate(args: GenerateArgs, model: &impl Model) -> String {
    let GenerateArgs {prompt, tokenizer, max_tokens, gpus, eos_token_id, num_layers} = args;

    let mut kv_cache = KVCache::new(num_layers);

    let input_ids = tokenizer.encode(prompt);

    let last_generated = initial_fill(&input_ids, model, &mut kv_cache, gpus);

    let mut all_ids = input_ids.clone();
    all_ids.push(last_generated);

    for _ in 0..max_tokens {
        let res = generate_single_pass(&mut all_ids, model, eos_token_id, &mut kv_cache, gpus);
        if res {
            break;
        }
    }

    let res = tokenizer.decode(&all_ids);

    res
}