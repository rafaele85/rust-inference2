use cudarc::driver::DeviceSlice;
use crate::root::activation::silu;
use crate::root::attention::{mla_attention, MLAAttentionArgs};
use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::config::{FFNConfig, MLAConfig, ModelDimConfig, MoeConfig, NormConfig};
use crate::root::gpu::{get_default_gpu, get_gpu, Gpu};
use crate::root::kv_cache::KVCache;
use crate::root::layer::{FFNLayer, MLAKVLayer, MOELayer, SharedExpertLayer};
use crate::root::linear::{add_to_vector, hadamard};
use crate::root::logits::logits;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::model::{GenerateNextToken, Model};
use crate::root::moe::{moe_ffn, MoeFFN};
use crate::root::norm::rms_norm;
use crate::root::tokenizer::BpeTokenizer;

pub struct GenerateNextTokenArgs<'a> {
    pub token_id: usize,
    pub position: usize,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>,
}
fn initial_fill<TM: GenerateNextToken>(input_ids: &Vec<u32>, model: &TM, kv_cache: &mut KVCache, gpus: &mut Vec<Gpu>) -> u32 {
    let mut last_generated = 0;

    for i in 0..input_ids.len() {
        let token_id = input_ids[i] as usize;
        last_generated = model.generate_next_token(GenerateNextTokenArgs {token_id, position: i, kv_cache, gpus});
    }

    last_generated as u32
}

fn generate_single_pass<TM: GenerateNextToken>(all_ids: &mut Vec<u32>, model: &TM, eos_token_id: &Vec<u32>, kv_cache: &mut KVCache, gpus: &mut Vec<Gpu>) -> bool {
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

pub fn generate<TM: GenerateNextToken>(args: GenerateArgs, model: &TM) -> String {
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


pub struct ApplyLayerArgs<'a, TLayer: SharedExpertLayer + FFNLayer + MOELayer, TConfig: ModelDimConfig + NormConfig + MoeConfig + FFNConfig> {
    pub hidden: &'a mut Vecf32,
    pub layer: &'a TLayer,
    pub config: &'a TConfig,
    pub kv_cache: &'a mut KVCache,
    pub layer_no: usize,
    pub position: usize,
    pub gpu: &'a mut Gpu,
}

pub fn apply_layer<TLayer: SharedExpertLayer + FFNLayer + MOELayer + MLAKVLayer, TConfig: ModelDimConfig + NormConfig + MLAConfig + MoeConfig + FFNConfig>(
    args: ApplyLayerArgs<TLayer, TConfig>
) {
    let ApplyLayerArgs { layer, config, kv_cache, layer_no, position, mut gpu, hidden} = args;

    let normed = rms_norm(hidden, layer.get_input_norm_w(), config.get_rms_norm_eps());

    mla_attention(MLAAttentionArgs {
        gpu: &mut gpu, hidden, kv_cache, layer_no, position, normed: &normed,
        layer,
        config,
    });

    if let (Some(router_w), Some(experts)) = (&layer.get_expert_router_w(), &layer.get_experts()) {
        let normed2 = rms_norm(hidden, &layer.get_post_norm_w(), config.get_rms_norm_eps());
        moe_ffn(MoeFFN {
            hidden, normed: &normed2, router_w, experts, config,
            gpu,
        });
        shared_experts(SharedExpertsArgs {hidden, normed: &normed2, layer, config, gpu});
    } else {
        if let (Some(gate_proj_w), Some(up_proj_w), Some(down_proj_w)) = (&layer.get_gate_proj_w(), &layer.get_up_proj_w(), &layer.get_down_proj_w()) {
            ffn(hidden, layer, gate_proj_w, up_proj_w, down_proj_w, config, gpu);
        }
    }
}



fn ffn<TLayer: FFNLayer, TConfig: MoeConfig + FFNConfig + NormConfig + MLAConfig + ModelDimConfig>(
    hidden: &mut Vecf32, layer: &TLayer, gate_proj_w: &WeightF16, up_proj_w: &WeightF16, down_proj_w: &WeightF16, config: &TConfig, gpu: &mut Gpu
) {
    let normed2 = rms_norm(&hidden, &layer.get_post_norm_w(), config.get_rms_norm_eps());

    let gate = matmul_transposed_f16(&normed2, &gate_proj_w, config.get_intermediate_size(), gpu);

    let up = matmul_transposed_f16(&normed2, &up_proj_w, config.get_intermediate_size(), gpu);

    let silu_gate = silu(&gate);

    let ffn_act = hadamard(silu_gate, up, config.get_intermediate_size());

    let ffn_out = matmul_transposed_f16(&ffn_act, &down_proj_w, config.get_hidden_size(), gpu);

    add_to_vector(hidden, &ffn_out);
}

struct SharedExpertsArgs<'a, TLayer: SharedExpertLayer, TConfig: MoeConfig + ModelDimConfig> {
    hidden: &'a mut Vecf32,
    normed: &'a Vecf32,
    layer: &'a TLayer,
    config: &'a TConfig,
    gpu: &'a mut Gpu,
}
fn shared_experts<TLayer: SharedExpertLayer, TConfig: MoeConfig + ModelDimConfig>(args: SharedExpertsArgs<TLayer, TConfig>) {
    let SharedExpertsArgs {hidden, normed, layer, config, gpu} = args;

    if let (Some(gate_w), Some(up_w), Some(down_w)) = (&layer.get_shared_gate_proj_w(), &layer.get_shared_up_proj_w(), &layer.get_shared_down_proj_w()) {
        let shared_intermediate = config.get_n_shared_experts() * config.get_moe_intermediate_size();
        if shared_intermediate == 0 {
            eprintln!("Warning: shared expert weights present but n_shared_experts * moe_intermediate_size = 0, skipping");
            return;
        }
        let gate = matmul_transposed_f16(normed, gate_w, shared_intermediate, gpu);
        let silu_gate = silu(&gate);
        let up = matmul_transposed_f16(normed, up_w, shared_intermediate, gpu);
        let ffn_act = hadamard(silu_gate, up, shared_intermediate);
        let ffn_out = matmul_transposed_f16(&ffn_act, down_w, config.get_hidden_size(), gpu);

        add_to_vector(hidden, &ffn_out);
    }
}


pub struct GenerateTokenArgs<'a, TLayer: SharedExpertLayer + FFNLayer + MOELayer + MLAKVLayer, TConfig: ModelDimConfig + NormConfig + MoeConfig + FFNConfig + MLAConfig, TM: Model> {
    pub token_id: usize,
    pub position: usize,
    pub config: &'a TConfig,
    pub layers: &'a Vec<TLayer>,
    pub kv_cache: &'a mut KVCache,
    pub gpus: &'a mut Vec<Gpu>,
    pub model: &'a TM,
}

pub fn generate_token<TLayer: SharedExpertLayer + FFNLayer + MOELayer + MLAKVLayer, TConfig: ModelDimConfig + NormConfig + MoeConfig + FFNConfig + MLAConfig, TM: Model>(
    args: GenerateTokenArgs<TLayer, TConfig, TM>
) -> usize {
    let GenerateTokenArgs {token_id, position, model, config, layers, kv_cache, mut gpus} = args;
    let token_start = token_id * config.get_hidden_size();

    let embed_token = &model.get_embed_weights()[token_start..token_start + config.get_hidden_size()];

    let mut hidden = embed_token.to_vec();

    for i in 0..config.get_num_hidden_layers() {
        let gpu = get_gpu(gpus, i, config.get_num_hidden_layers());
        apply_layer(ApplyLayerArgs {
            hidden: &mut hidden,
            layer: &layers[i],
            config,
            kv_cache,
            layer_no: i,
            position,
            gpu,
        });
    }

    let default_gpu = get_default_gpu(&mut gpus, "logits");
    let output_token_id = logits(hidden, model.get_final_norm_w(), model.get_lm_head_w(), config.get_vocab_size(), config.get_rms_norm_eps(), default_gpu);

    output_token_id
}


