use crate::gpu::Gpu;
use crate::kv_cache::KVCache;
use crate::tokenizer::BpeTokenizer;

pub struct GenerateNextTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub kv_cache: &'a mut KVCache,
    pub gpu: &'a Gpu
}
pub trait Model {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize;
}

fn initial_fill(input_ids: &Vec<u32>, model: &impl Model, kv_cache: &mut KVCache, gpu: &Gpu) -> u32 {
    let mut last_generated = 0;

    for i in 0..input_ids.len() {
        let token_id = input_ids[i] as usize;
        last_generated = model.generate_next_token(GenerateNextTokenArgs {token_id, position: i, kv_cache, gpu});
    }

    last_generated as u32
}

fn generate_single_pass(all_ids: &mut Vec<u32>, model: &impl Model, eos_token_id: u32, kv_cache: &mut KVCache, gpu: &Gpu) -> bool {
    let last_id = *all_ids.last().unwrap();
    if last_id == eos_token_id {
        return true;
    }
    let pos = all_ids.len() - 1;
    let top_id = model.generate_next_token(GenerateNextTokenArgs {token_id: last_id as usize, position: pos, kv_cache, gpu});
    all_ids.push(top_id as u32);

    false
}

pub struct GenerateArgs<'a> {
    pub prompt: &'a str,
    pub tokenizer: &'a BpeTokenizer,
    pub max_tokens: usize,
    pub kv_cache: &'a mut KVCache,
    pub gpu: &'a Gpu,
    pub eos_token_id: u32,
}

pub fn generate(args: GenerateArgs, model: &impl Model) -> String {
    let GenerateArgs {prompt, tokenizer, max_tokens, kv_cache, gpu, eos_token_id} = args;
    let input_ids = tokenizer.encode(prompt);

    println!("input ids: {:?}", input_ids);

    let last_generated = initial_fill(&input_ids, model, kv_cache, gpu);

    let mut all_ids = input_ids.clone();
    all_ids.push(last_generated);

    for _ in 0..max_tokens {
        let res = generate_single_pass(&mut all_ids, model, eos_token_id, kv_cache, gpu);
        if res {
            break;
        }
    }

    let res = tokenizer.decode(&all_ids);

    res
}