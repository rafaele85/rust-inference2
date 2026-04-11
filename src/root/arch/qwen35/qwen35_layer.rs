use std::time::Instant;
use cudarc::driver::{LaunchAsync};
use half::f16;
use crate::root::arch::qwen35::qwen35_config::{Qwen35Config, TextConfig};
use crate::root::arch::safetensors::{load_mlx_affine, load_scales, load_tensor_to_f32, load_tensor_u32, SafeTensor};
use crate::root::common_types::{Arrf32, Vecf32, WeightF16, WeightU32};
use crate::root::gpu::Gpu;

pub struct PackedExperts {
    pub gate_proj_packed: WeightU32,
    pub gate_proj_scales: WeightF16,
    pub gate_proj_biases: WeightF16,
    pub up_proj_packed: WeightU32,
    pub up_proj_scales: WeightF16,
    pub up_proj_biases: WeightF16,
    pub down_proj_packed: WeightU32,
    pub down_proj_scales: WeightF16,
    pub down_proj_biases: WeightF16,
    pub num_experts: usize,
    pub moe_intermediate: usize,
    pub hidden_size: usize,
    pub group_size: usize,
}


impl PackedExperts {
    pub(crate) fn matmul(
        &self,
        input: &Arrf32,
        packed: &WeightU32,
        scales: &WeightF16,
        biases: &WeightF16,
        expert_idx: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        gpu: &mut Gpu
    ) -> Vecf32 {
        let packed_cols = in_dim / 8;
        let scale_cols = in_dim / group_size;
        let packed_offset = expert_idx * out_dim * packed_cols;
        let scale_offset = expert_idx * out_dim * scale_cols;

        gpu.scratch.ensure_expert_f16(&gpu.dev, out_dim * in_dim);

        let func = gpu.dev.get_func("mlx_dequant", "mlx_dequant_4bit").unwrap();
        let total_packed = (out_dim * packed_cols) as u32;
        let block_size = 256u32;
        let grid_size = (total_packed + block_size - 1) / block_size;

        unsafe {
            func.launch(
                cudarc::driver::LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &packed.slice(packed_offset..),
                    &scales.slice(scale_offset..),
                    &biases.slice(scale_offset..),
                    &gpu.scratch.expert_f16,
                    out_dim as i32,
                    packed_cols as i32,
                    scale_cols as i32,
                    group_size as i32,
                ),
            ).unwrap();
        }

        let expert_w = std::mem::replace(
            &mut gpu.scratch.expert_f16,
            gpu.dev.alloc_zeros::<f16>(1).unwrap(),
        );

        let result = gpu.matmul_transposed_f16(input, &expert_w, in_dim, out_dim);

        gpu.scratch.expert_f16 = expert_w;

        result
    }
}

pub struct Qwen35MoeWeights {
    pub router_w: Vecf32,
    pub shared_expert_gate_w: Vecf32,
    pub experts: PackedExperts,
    pub shared_gate_proj_w: WeightF16,
    pub shared_up_proj_w: WeightF16,
    pub shared_down_proj_w: WeightF16,
}
pub struct Qwen35FullAttnLayer {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,

    // Attention (q_proj includes output gate, so 2x attn_dim)
    pub q_proj_w: WeightF16,
    pub k_proj_w: WeightF16,
    pub v_proj_w: WeightF16,
    pub o_proj_w: WeightF16,
    pub q_norm_w: Vecf32,
    pub k_norm_w: Vecf32,

    pub moe: Qwen35MoeWeights,
}

impl Qwen35FullAttnLayer {
    pub fn load(st: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu, config: &Qwen35Config) -> Self {
        let prefix = format!("language_model.model.layers.{}", layer_no);

        // Norms (BF16, not quantized)
        let input_norm_w = load_tensor_to_f32(st, &format!("{}.input_layernorm.weight", prefix));
        let post_norm_w = load_tensor_to_f32(st, &format!("{}.post_attention_layernorm.weight", prefix));

        // Attention projections (quantized)
        let bits = config.bits();
        let group_size = config.group_size();
        let q_proj_w = load_and_upload(st, gpu, &prefix, "self_attn.q_proj", bits, group_size, "q_proj", layer_no);
        let k_proj_w = load_and_upload(st, gpu, &prefix, "self_attn.k_proj", bits, group_size, "k_proj", layer_no);
        let v_proj_w = load_and_upload(st, gpu, &prefix, "self_attn.v_proj", bits, group_size, "v_proj", layer_no);
        let o_proj_w = load_and_upload(st, gpu, &prefix, "self_attn.o_proj", bits, group_size, "o_proj", layer_no);


        // QK norms (BF16)
        let q_norm_w = load_tensor_to_f32(st, &format!("{}.self_attn.q_norm.weight", prefix));
        let k_norm_w = load_tensor_to_f32(st, &format!("{}.self_attn.k_norm.weight", prefix));

        // MoE
        let moe = load_moe_weights(st, gpu, &prefix, bits, group_size, layer_no, &config.text_config);

        Self {
            input_norm_w, post_norm_w,
            q_proj_w, k_proj_w, v_proj_w, o_proj_w,
            q_norm_w, k_norm_w,
            moe
        }
    }
}

fn load_and_upload(
    st: &mut SafeTensor, gpu: &mut Gpu, prefix: &str, name: &str,
    bits: usize, group_size: usize, label: &str, layer_no: usize,
) -> WeightF16 {
    let w = load_mlx_affine(
        st,
        &format!("{}.{}.weight", prefix, name),
        &format!("{}.{}.scales", prefix, name),
        &format!("{}.{}.biases", prefix, name),
        bits, group_size,
    );
    gpu.upload_f16_from_f32(&w, &format!("{}_{}", label, layer_no))
}


pub struct Qwen35LinearAttnLayer {
    pub input_norm_w: Vecf32,
    pub post_norm_w: Vecf32,

    // GatedDeltaNet projections
    pub in_proj_qkv_w: WeightF16,  // [12288, 3072] -> q(2048) + k(2048) + v(8192)
    pub in_proj_a_w: WeightF16,    // [64, 3072] -> alpha gate per v_head
    pub in_proj_b_w: WeightF16,    // [64, 3072] -> beta gate per v_head
    pub in_proj_z_w: WeightF16,    // [8192, 3072] -> output gate (v_dim)
    pub out_proj_w: WeightF16,     // [3072, 8192] -> project back to hidden

    pub a_log: Vecf32,             // [64] decay parameter
    pub dt_bias: Vecf32,           // [64] timestep bias
    pub conv1d_w: Vecf32,          // [12288, 4, 1] depthwise conv
    pub norm_w: Vecf32,            // [128] group norm per head

    pub moe: Qwen35MoeWeights,
}

impl Qwen35LinearAttnLayer {
    pub fn load(st: &mut SafeTensor, layer_no: usize, gpu: &mut Gpu, config: &Qwen35Config) -> Self {
        let prefix = format!("language_model.model.layers.{}", layer_no);

        let input_norm_w = load_tensor_to_f32(st, &format!("{}.input_layernorm.weight", prefix));
        let post_norm_w = load_tensor_to_f32(st, &format!("{}.post_attention_layernorm.weight", prefix));
        
        // GatedDeltaNet projections (quantized)
        let bits = config.bits();
        let group_size = config.group_size();

        let in_proj_qkv_w = load_and_upload(st, gpu, &prefix, "linear_attn.in_proj_qkv", bits, group_size, "qkv", layer_no);

        let in_proj_a_w = load_and_upload(st, gpu, &prefix, "linear_attn.in_proj_a", bits, group_size, "a", layer_no);
        let in_proj_b_w = load_and_upload(st, gpu, &prefix, "linear_attn.in_proj_b", bits, group_size, "b", layer_no);
        let in_proj_z_w = load_and_upload(st, gpu, &prefix, "linear_attn.in_proj_z", bits, group_size, "z", layer_no);
        let out_proj_w = load_and_upload(st, gpu, &prefix, "linear_attn.out_proj", bits, group_size, "out", layer_no);

        // Non-quantized
        let a_log = load_tensor_to_f32(st, &format!("{}.linear_attn.A_log", prefix));
        let dt_bias = load_tensor_to_f32(st, &format!("{}.linear_attn.dt_bias", prefix));
        let conv1d_w = load_tensor_to_f32(st, &format!("{}.linear_attn.conv1d.weight", prefix));
        let norm_w = load_tensor_to_f32(st, &format!("{}.linear_attn.norm.weight", prefix));

        // MoE (same as full attention)
        let moe = load_moe_weights(st, gpu, &prefix, bits, group_size, layer_no, &config.text_config);

        Self {
            input_norm_w, post_norm_w,
            in_proj_qkv_w, in_proj_a_w, in_proj_b_w, in_proj_z_w, out_proj_w,
            a_log, dt_bias, conv1d_w, norm_w,
            moe,
        }
    }
}



fn upload_packed_proj(
    st: &mut SafeTensor, gpu: &mut Gpu, prefix: &str, proj_name: &str, layer_no: usize,
) -> (WeightU32, WeightF16, WeightF16) {
    let packed = load_tensor_u32(st, &format!("{}.mlp.switch_mlp.{}.weight", prefix, proj_name));
    let scales = load_scales(st, &format!("{}.mlp.switch_mlp.{}.scales", prefix, proj_name));
    let biases = load_scales(st, &format!("{}.mlp.switch_mlp.{}.biases", prefix, proj_name));

    let d_packed = gpu.upload_u32(&packed, &format!("exp_{}_{}", proj_name, layer_no));
    let d_scales = gpu.upload_f16_from_f32(&scales, &format!("exp_{}_sc_{}", proj_name, layer_no));
    let d_biases = gpu.upload_f16_from_f32(&biases, &format!("exp_{}_bi_{}", proj_name, layer_no));

    (d_packed, d_scales, d_biases)
}

pub fn load_moe_weights(
    st: &mut SafeTensor, gpu: &mut Gpu, prefix: &str,
    bits: usize, group_size: usize, layer_no: usize,
    tc: &TextConfig,
) -> Qwen35MoeWeights {
    // Router (8-bit quantized, keep on CPU)
    let router_w = load_mlx_affine(st,
                                   &format!("{}.mlp.gate.weight", prefix),
                                   &format!("{}.mlp.gate.scales", prefix),
                                   &format!("{}.mlp.gate.biases", prefix),
                                   8, group_size);

    // Shared expert gate (8-bit, keep on CPU)
    let shared_expert_gate_w = load_mlx_affine(st,
                                               &format!("{}.mlp.shared_expert_gate.weight", prefix),
                                               &format!("{}.mlp.shared_expert_gate.scales", prefix),
                                               &format!("{}.mlp.shared_expert_gate.biases", prefix),
                                               8, group_size);

    // Fused experts — upload packed directly to GPU
    let (gp, gs, gb) = upload_packed_proj(st, gpu, prefix, "gate_proj", layer_no);
    let (up, us, ub) = upload_packed_proj(st, gpu, prefix, "up_proj", layer_no);
    let (dp, ds, db) = upload_packed_proj(st, gpu, prefix, "down_proj", layer_no);

    let experts = PackedExperts {
        gate_proj_packed: gp, gate_proj_scales: gs, gate_proj_biases: gb,
        up_proj_packed: up, up_proj_scales: us, up_proj_biases: ub,
        down_proj_packed: dp, down_proj_scales: ds, down_proj_biases: db,
        num_experts: tc.num_experts, moe_intermediate: tc.moe_intermediate_size, hidden_size: tc.hidden_size,
        group_size,
    };

    // Shared expert (single expert, small, dequant and upload as f16)
    let shared_gate_proj_w = load_and_upload(st, gpu, prefix, "mlp.shared_expert.gate_proj", bits, group_size, "sh_gate", layer_no);
    let shared_up_proj_w = load_and_upload(st, gpu, prefix, "mlp.shared_expert.up_proj", bits, group_size, "sh_up", layer_no);
    let shared_down_proj_w = load_and_upload(st, gpu, prefix, "mlp.shared_expert.down_proj", bits, group_size, "sh_down", layer_no);

    Qwen35MoeWeights {
        router_w, shared_expert_gate_w, experts,
        shared_gate_proj_w, shared_up_proj_w, shared_down_proj_w,
    }
}

