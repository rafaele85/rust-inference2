use crate::root::tokenizer::BpeTokenizer;

use cudarc::driver::{CudaSlice, DeviceSlice, LaunchAsync};
use cudarc::nvrtc::Ptx;
use half::f16;
use crate::root::activation::{sigmoid, silu, softmax};
use crate::root::arch::llama::llama::LlamaModel;
use crate::root::arch::qwen35::qwen35::{Qwen35Layer, Qwen35Model};
use crate::root::arch::qwen35::qwen35_attention::{qwen35_apply_rope, qwen35_compute_qkv, qwen35_full_attention, qwen35_gate_and_project, qwen35_gqa_attention};
use crate::root::arch::qwen35::qwen35_config::Qwen35Config;
use crate::root::arch::qwen35::qwen35_layer::{load_moe_weights, Qwen35FullAttnLayer, Qwen35LinearAttnLayer};
use crate::root::arch::qwen35::qwen35_linear_attention::{qwen35_compute_delta_gates, qwen35_conv1d_update, qwen35_linear_attention, qwen35_linear_attn_output, qwen35_recurrent_update, qwen35_split_qkv};
use crate::root::arch::qwen35::qwen35_moe::qwen35_moe_ffn;
use crate::root::arch::qwen35::qwen35_transformer::qwen35_generate_token;
use crate::root::arch::safetensors::{load_mlx_affine, load_scales, load_tensor_to_f32, load_tensor_u32, SafeTensor};
use crate::root::generate::{generate, GenerateArgs};
use crate::root::gpu::Gpu;
use crate::root::kv_cache::KVCache;
use crate::root::linear::add_to_vector;
use crate::root::matmul::matmul_transposed_f16;
use crate::root::moe::{compute_scores, get_top_scores};
use crate::root::norm::{rms_norm, rms_norm_zero_centered};


const PATH: &str = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit";


pub fn qwen35_test_config(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    println!("hidden_size: {}", tc.hidden_size);
    println!("num_layers: {}", tc.num_hidden_layers);
    println!("num_experts: {}", tc.num_experts);
    println!("layer_types: {} linear, {} full",
             tc.layer_types.iter().filter(|t| *t == "linear_attention").count(),
             tc.layer_types.iter().filter(|t| *t == "full_attention").count(),
    );
    println!("bits: {}, group_size: {}", config.bits(), config.group_size());
    println!("eos_token_id: {:?}", config.eos_token_id);
    println!("✅ Config parsed successfully");
}

pub fn qwen35_test_weights(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let st = SafeTensor::new(path);

    // Check a few key tensors
    let keys = vec![
        "language_model.model.embed_tokens.weight",
        "language_model.model.embed_tokens.scales",
        "language_model.model.embed_tokens.biases",
        "language_model.model.norm.weight",
        "language_model.lm_head.weight",
        // Layer 0 linear attention
        "language_model.model.layers.0.linear_attn.in_proj_qkv.weight",
        "language_model.model.layers.0.linear_attn.in_proj_qkv.scales",
        "language_model.model.layers.0.linear_attn.A_log",
        "language_model.model.layers.0.linear_attn.dt_bias",
        "language_model.model.layers.0.linear_attn.conv1d.weight",
        // Layer 0 MoE
        "language_model.model.layers.0.mlp.gate.weight",
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
        // Layer 3 full attention
        "language_model.model.layers.3.self_attn.q_proj.weight",
        "language_model.model.layers.3.self_attn.k_proj.weight",
        "language_model.model.layers.3.self_attn.q_norm.weight",
    ];

    for key in keys {
        if let Some(entry) = st.entries.get(key) {
            println!("{:70} dtype={:5} shape={:?}", key, entry.dtype, entry.shape);
        } else {
            println!("{:70} NOT FOUND", key);
        }
    }
    println!("✅ Weight inspection done");
}

pub fn qwen35_test_load_full_attn(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let gpu = &mut gpus[0];
    let bits = config.bits();
    let gs = config.group_size();

    let prefix = "language_model.model.layers.3";

    // Norms (BF16, not quantized)
    let input_norm = load_tensor_to_f32(&mut st, &format!("{}.input_layernorm.weight", prefix));
    let post_norm = load_tensor_to_f32(&mut st, &format!("{}.post_attention_layernorm.weight", prefix));
    println!("input_norm len={}, post_norm len={}", input_norm.len(), post_norm.len());


    let q_entry = &st.entries[&format!("{}.self_attn.q_proj.weight", prefix)];
    let k_entry = &st.entries[&format!("{}.self_attn.k_proj.weight", prefix)];
    let v_entry = &st.entries[&format!("{}.self_attn.v_proj.weight", prefix)];
    let o_entry = &st.entries[&format!("{}.self_attn.o_proj.weight", prefix)];
    println!("q_proj shape: {:?}", q_entry.shape);
    println!("k_proj shape: {:?}", k_entry.shape);
    println!("v_proj shape: {:?}", v_entry.shape);
    println!("o_proj shape: {:?}", o_entry.shape);



    // Attention weights (quantized)
    let q_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.q_proj.weight", prefix),
                                 &format!("{}.self_attn.q_proj.scales", prefix),
                                 &format!("{}.self_attn.q_proj.biases", prefix),
                                 bits, gs);
    let k_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.k_proj.weight", prefix),
                                 &format!("{}.self_attn.k_proj.scales", prefix),
                                 &format!("{}.self_attn.k_proj.biases", prefix),
                                 bits, gs);
    let v_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.v_proj.weight", prefix),
                                 &format!("{}.self_attn.v_proj.scales", prefix),
                                 &format!("{}.self_attn.v_proj.biases", prefix),
                                 bits, gs);
    let o_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.o_proj.weight", prefix),
                                 &format!("{}.self_attn.o_proj.scales", prefix),
                                 &format!("{}.self_attn.o_proj.biases", prefix),
                                 bits, gs);

    let tc = &config.text_config;
    let attn_dim = tc.num_attention_heads * tc.head_dim; // 32 * 256 = 8192
    let kv_dim = tc.num_key_value_heads * tc.head_dim;   // 2 * 256 = 512



    println!("q_proj len={}, expected={}", q_proj.len(), attn_dim * tc.hidden_size);
    println!("k_proj len={}, expected={}", k_proj.len(), kv_dim * tc.hidden_size);
    println!("v_proj len={}, expected={}", v_proj.len(), kv_dim * tc.hidden_size);
    println!("o_proj len={}, expected={}", o_proj.len(), tc.hidden_size * attn_dim);

    assert_eq!(q_proj.len(), attn_dim * tc.hidden_size);
    assert_eq!(k_proj.len(), kv_dim * tc.hidden_size);
    assert_eq!(v_proj.len(), kv_dim * tc.hidden_size);
    assert_eq!(o_proj.len(), tc.hidden_size * attn_dim);

    // QK norms
    let q_norm = load_tensor_to_f32(&mut st, &format!("{}.self_attn.q_norm.weight", prefix));
    let k_norm = load_tensor_to_f32(&mut st, &format!("{}.self_attn.k_norm.weight", prefix));
    println!("q_norm len={}, k_norm len={}", q_norm.len(), k_norm.len());
    assert_eq!(q_norm.len(), tc.head_dim);
    assert_eq!(k_norm.len(), tc.head_dim);

    println!("✅ Full attention layer 3 weights loaded correctly");


}

pub fn qwen35_test_load_full_attn2(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let gpu = &mut gpus[0];
    let bits = config.bits();
    let gs = config.group_size();

    let prefix = "language_model.model.layers.3";

    // Norms (BF16, not quantized)
    let input_norm = load_tensor_to_f32(&mut st, &format!("{}.input_layernorm.weight", prefix));
    let post_norm = load_tensor_to_f32(&mut st, &format!("{}.post_attention_layernorm.weight", prefix));
    println!("input_norm len={}, post_norm len={}", input_norm.len(), post_norm.len());

    // Check shapes
    let q_entry = &st.entries[&format!("{}.self_attn.q_proj.weight", prefix)];
    let k_entry = &st.entries[&format!("{}.self_attn.k_proj.weight", prefix)];
    let v_entry = &st.entries[&format!("{}.self_attn.v_proj.weight", prefix)];
    let o_entry = &st.entries[&format!("{}.self_attn.o_proj.weight", prefix)];
    println!("q_proj shape: {:?}", q_entry.shape);
    println!("k_proj shape: {:?}", k_entry.shape);
    println!("v_proj shape: {:?}", v_entry.shape);
    println!("o_proj shape: {:?}", o_entry.shape);

    // Attention weights (quantized)
    let q_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.q_proj.weight", prefix),
                                 &format!("{}.self_attn.q_proj.scales", prefix),
                                 &format!("{}.self_attn.q_proj.biases", prefix),
                                 bits, gs);
    let k_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.k_proj.weight", prefix),
                                 &format!("{}.self_attn.k_proj.scales", prefix),
                                 &format!("{}.self_attn.k_proj.biases", prefix),
                                 bits, gs);
    let v_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.v_proj.weight", prefix),
                                 &format!("{}.self_attn.v_proj.scales", prefix),
                                 &format!("{}.self_attn.v_proj.biases", prefix),
                                 bits, gs);
    let o_proj = load_mlx_affine(&mut st,
                                 &format!("{}.self_attn.o_proj.weight", prefix),
                                 &format!("{}.self_attn.o_proj.scales", prefix),
                                 &format!("{}.self_attn.o_proj.biases", prefix),
                                 bits, gs);

    let tc = &config.text_config;
    let attn_dim = tc.num_attention_heads * tc.head_dim;  // 32 * 256 = 8192
    let kv_dim = tc.num_key_value_heads * tc.head_dim;    // 2 * 256 = 512
    let q_proj_dim = attn_dim * 2;                        // 16384 (q + gate)

    println!("q_proj len={}, expected={}", q_proj.len(), q_proj_dim * tc.hidden_size);
    println!("k_proj len={}, expected={}", k_proj.len(), kv_dim * tc.hidden_size);
    println!("v_proj len={}, expected={}", v_proj.len(), kv_dim * tc.hidden_size);
    println!("o_proj len={}, expected={}", o_proj.len(), tc.hidden_size * attn_dim);

    assert_eq!(q_proj.len(), q_proj_dim * tc.hidden_size);
    assert_eq!(k_proj.len(), kv_dim * tc.hidden_size);
    assert_eq!(v_proj.len(), kv_dim * tc.hidden_size);
    assert_eq!(o_proj.len(), tc.hidden_size * attn_dim);

    // QK norms
    let q_norm = load_tensor_to_f32(&mut st, &format!("{}.self_attn.q_norm.weight", prefix));
    let k_norm = load_tensor_to_f32(&mut st, &format!("{}.self_attn.k_norm.weight", prefix));
    println!("q_norm len={}, k_norm len={}", q_norm.len(), k_norm.len());
    assert_eq!(q_norm.len(), tc.head_dim);
    assert_eq!(k_norm.len(), tc.head_dim);

    println!("✅ Full attention layer 3 weights loaded correctly");
}


pub fn qwen35_test_load_linear_attn(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let bits = config.bits();
    let gs = config.group_size();
    let tc = &config.text_config;

    let prefix = "language_model.model.layers.0";

    // Print all linear_attn shapes
    let keys = vec![
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_a",
        "linear_attn.in_proj_b",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    ];
    for key in &keys {
        let wk = format!("{}.{}.weight", prefix, key);
        let entry = &st.entries[&wk];
        let pack_size = 32 / bits;
        let out_dim = entry.shape[0];
        let in_dim = entry.shape[1] * pack_size;
        println!("{:40} shape={:?} -> dequant [{}, {}]", key, entry.shape, out_dim, in_dim);
    }

    // Non-quantized tensors
    let non_quant = vec![
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
    ];
    for key in &non_quant {
        let wk = format!("{}.{}", prefix, key);
        let entry = &st.entries[&wk];
        println!("{:40} shape={:?} dtype={}", key, entry.shape, entry.dtype);
    }

    // Dequant all quantized linear_attn weights
    let in_proj_qkv = load_mlx_affine(&mut st,
                                      &format!("{}.linear_attn.in_proj_qkv.weight", prefix),
                                      &format!("{}.linear_attn.in_proj_qkv.scales", prefix),
                                      &format!("{}.linear_attn.in_proj_qkv.biases", prefix), bits, gs);
    let in_proj_a = load_mlx_affine(&mut st,
                                    &format!("{}.linear_attn.in_proj_a.weight", prefix),
                                    &format!("{}.linear_attn.in_proj_a.scales", prefix),
                                    &format!("{}.linear_attn.in_proj_a.biases", prefix), bits, gs);
    let in_proj_b = load_mlx_affine(&mut st,
                                    &format!("{}.linear_attn.in_proj_b.weight", prefix),
                                    &format!("{}.linear_attn.in_proj_b.scales", prefix),
                                    &format!("{}.linear_attn.in_proj_b.biases", prefix), bits, gs);
    let in_proj_z = load_mlx_affine(&mut st,
                                    &format!("{}.linear_attn.in_proj_z.weight", prefix),
                                    &format!("{}.linear_attn.in_proj_z.scales", prefix),
                                    &format!("{}.linear_attn.in_proj_z.biases", prefix), bits, gs);
    let out_proj = load_mlx_affine(&mut st,
                                   &format!("{}.linear_attn.out_proj.weight", prefix),
                                   &format!("{}.linear_attn.out_proj.scales", prefix),
                                   &format!("{}.linear_attn.out_proj.biases", prefix), bits, gs);

    // Non-quantized
    let a_log = load_tensor_to_f32(&mut st, &format!("{}.linear_attn.A_log", prefix));
    let dt_bias = load_tensor_to_f32(&mut st, &format!("{}.linear_attn.dt_bias", prefix));
    let conv1d_w = load_tensor_to_f32(&mut st, &format!("{}.linear_attn.conv1d.weight", prefix));
    let norm_w = load_tensor_to_f32(&mut st, &format!("{}.linear_attn.norm.weight", prefix));

    println!("\nin_proj_qkv len={}", in_proj_qkv.len());
    println!("in_proj_a len={}", in_proj_a.len());
    println!("in_proj_b len={}", in_proj_b.len());
    println!("in_proj_z len={}", in_proj_z.len());
    println!("out_proj len={}", out_proj.len());
    println!("A_log len={}", a_log.len());
    println!("dt_bias len={}", dt_bias.len());
    println!("conv1d_w len={}", conv1d_w.len());
    println!("norm_w len={}", norm_w.len());

    // Expected dims from config
    let k_dim = tc.linear_num_key_heads * tc.linear_key_head_dim;     // 16 * 128 = 2048
    let v_dim = tc.linear_num_value_heads * tc.linear_value_head_dim; // 64 * 128 = 8192
    println!("\nExpected: q=?, k_dim={}, v_dim={}, hidden={}", k_dim, v_dim, tc.hidden_size);

    println!("✅ Linear attention layer 0 weights loaded");
}


pub fn qwen35_test_load_layer(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));

    println!("Loading full attention layer 3...");
    let layer = Qwen35FullAttnLayer::load(&mut st, 3, &mut gpus[0], &config);
    println!("input_norm_w len={}", layer.input_norm_w.len());
    println!("q_norm_w len={}", layer.q_norm_w.len());
    println!("router_w len={}", layer.moe.router_w.len());
    println!("✅ Full attention layer 3 loaded");
}


pub fn qwen35_test_load_both_layers(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let bits = config.bits();
    let gs = config.group_size();

    println!("Loading linear attention layer 0...");
    let linear = Qwen35LinearAttnLayer::load(&mut st, 0, &mut gpus[0], &config);
    println!("  a_log len={}", linear.a_log.len());
    println!("  dt_bias len={}", linear.dt_bias.len());
    println!("  conv1d_w len={}", linear.conv1d_w.len());
    println!("  norm_w len={}", linear.norm_w.len());
    println!("✅ Linear attention layer 0 loaded");

    println!("Loading full attention layer 3...");
    let full = Qwen35FullAttnLayer::load(&mut st, 3, &mut gpus[0],  &config);
    println!("  q_norm_w len={}", full.q_norm_w.len());
    println!("✅ Full attention layer 3 loaded");
}

pub fn qwen35_test_load_model(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);

    println!("Loading Qwen3.5 model...");
    let model = Qwen35Model::new(&mut st, gpus);
    println!("Layers loaded: {}", model.layers.len());
    println!("Embed len: {}", model.embed_weights.len());
    println!("Final norm len: {}", model.final_norm_w.len());

    let linear_count = model.layers.iter().filter(|l| matches!(l, Qwen35Layer::Linear(_))).count();
    let full_count = model.layers.iter().filter(|l| matches!(l, Qwen35Layer::Full(_))).count();
    println!("Linear layers: {}, Full layers: {}", linear_count, full_count);

    assert_eq!(model.layers.len(), 48);
    assert_eq!(linear_count, 36);
    assert_eq!(full_count, 12);
    println!("✅ Qwen3.5 model loaded successfully");
}


pub fn qwen35_test_upload_packed(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let gpu = &mut gpus[0];

    // Load raw packed u32 from safetensor
    let packed = load_tensor_u32(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
    println!("packed len: {} ({:.1} MB)", packed.len(), packed.len() as f64 * 4.0 / 1024.0 / 1024.0);

    // Upload to GPU
    let d_packed = gpu.upload_u32(&packed, "test_packed");
    println!("uploaded to GPU, device len: {}", d_packed.len());

    // Read back and verify
    let readback = gpu.dev.dtoh_sync_copy(&d_packed).unwrap();
    assert_eq!(readback.len(), packed.len());
    assert_eq!(&readback[..10], &packed[..10]);

    println!("✅ Packed u32 upload + readback verified");
}

pub fn qwen35_test_upload_scales(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let gpu = &mut gpus[0];

    let scales = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales");
    let biases = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases");
    println!("scales len: {} ({:.1} KB)", scales.len(), scales.len() as f64 * 4.0 / 1024.0);
    println!("biases len: {} ({:.1} KB)", biases.len(), biases.len() as f64 * 4.0 / 1024.0);

    // Upload as f16
    let d_scales = gpu.upload_f16_from_f32(&scales, "test_scales");
    let d_biases = gpu.upload_f16_from_f32(&biases, "test_biases");
    println!("scales on GPU len: {}", d_scales.len());
    println!("biases on GPU len: {}", d_biases.len());

    // Expected: [256, 1024, 48] flattened = 256 * 1024 * 48 = 12582912
    // group_size=64, in_dim=3072, scale_cols = 3072/64 = 48
    assert_eq!(d_scales.len(), 256 * 1024 * 48);
    assert_eq!(d_biases.len(), 256 * 1024 * 48);

    println!("✅ Scales + biases upload verified");
}



pub fn qwen35_test_dequant_kernel(gpus: &mut Vec<Gpu>) {
    let gpu = &mut gpus[0];

    let ptx_src = std::fs::read_to_string("../../../ptx/mlx_dequant_4bit.ptx").unwrap();
    gpu.dev.load_ptx(Ptx::from_src(ptx_src), "mlx_dequant", &["mlx_dequant_4bit"]).unwrap();
    // Synthetic test: 2 rows, 8 cols (1 packed u32 per row)
    // packed_cols=1, in_dim=8, scale_cols=1 (group_size=8 so 1 group)
    let packed: Vec<u32> = vec![
        0x76543210,  // row 0: nibbles 0,1,2,3,4,5,6,7
        0xFEDCBA98,  // row 1: nibbles 8,9,10,11,12,13,14,15
    ];
    let scales: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(1.0)];
    let biases: Vec<f16> = vec![f16::from_f32(0.0), f16::from_f32(0.0)];

    let d_packed = gpu.dev.htod_sync_copy(&packed).unwrap();
    let d_scales = gpu.dev.htod_sync_copy(&scales).unwrap();
    let d_biases = gpu.dev.htod_sync_copy(&biases).unwrap();
    let d_output: CudaSlice<f16> = gpu.dev.alloc_zeros::<f16>(16).unwrap();

    let func = gpu.dev.get_func("mlx_dequant", "mlx_dequant_4bit").unwrap();
    let out_dim: i32 = 2;
    let packed_cols: i32 = 1;
    let scale_cols: i32 = 1;
    let group_size: i32 = 8;
    let total_packed = 2u32;
    let block_size = 256u32;
    let grid_size = (total_packed + block_size - 1) / block_size;

    unsafe {
        func.launch(
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (&d_packed, &d_scales, &d_biases, &d_output, out_dim, packed_cols, scale_cols, group_size),
        ).unwrap();
    }

    let result: Vec<f16> = gpu.dev.dtoh_sync_copy(&d_output).unwrap();
    let result_f32: Vec<f32> = result.iter().map(|x| x.to_f32()).collect();
    println!("row 0: {:?}", &result_f32[..8]);
    println!("row 1: {:?}", &result_f32[8..16]);

    // With scale=1.0, bias=0.0, output should be the nibble values
    assert_eq!(result_f32[0], 0.0);
    assert_eq!(result_f32[1], 1.0);
    assert_eq!(result_f32[7], 7.0);
    assert_eq!(result_f32[8], 8.0);
    assert_eq!(result_f32[15], 15.0);

    println!("✅ Dequant kernel test passed");
}

pub fn qwen35_test_dequant_kernel_real(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let gpu = &mut gpus[0];

    let ptx = Ptx::from_file("ptx/mlx_dequant.ptx");
    gpu.dev.load_ptx(ptx, "mlx_dequant", &["mlx_dequant_4bit"]).unwrap();

    // Upload packed weights + scales + biases to GPU
    let packed = load_tensor_u32(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
    let scales = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales");
    let biases = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases");

    let d_packed = gpu.upload_u32(&packed, "packed");
    let d_scales = gpu.upload_f16_from_f32(&scales, "scales");
    let d_biases = gpu.upload_f16_from_f32(&biases, "biases");

    // Dequant expert 5 on GPU
    let expert_idx = 5usize;
    let out_dim = 1024;       // moe_intermediate_size
    let packed_cols = 384;    // hidden_size / 8
    let scale_cols = 48;      // hidden_size / group_size
    let in_dim = 3072;        // hidden_size

    // Offsets into the fused tensor for expert 5
    let packed_offset = expert_idx * out_dim * packed_cols;
    let scale_offset = expert_idx * out_dim * scale_cols;

    // Allocate scratch for one expert
    let d_output: CudaSlice<f16> = gpu.dev.alloc_zeros::<f16>(out_dim * in_dim).unwrap();

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
                &d_packed.slice(packed_offset..),
                &d_scales.slice(scale_offset..),
                &d_biases.slice(scale_offset..),
                &d_output,
                out_dim as i32,
                packed_cols as i32,
                scale_cols as i32,
                config.group_size() as i32,
            ),
        ).unwrap();
    }

    // Read back GPU result
    let gpu_result_f16: Vec<f16> = gpu.dev.dtoh_sync_copy(&d_output).unwrap();
    let gpu_result: Vec<f32> = gpu_result_f16.iter().map(|x| x.to_f32()).collect();

    // CPU reference: dequant the full tensor then slice expert 5
    let cpu_full = load_mlx_affine(&mut st,
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
                                   config.bits(), config.group_size());
    let expert_size = out_dim * in_dim;
    let cpu_expert = &cpu_full[expert_idx * expert_size..(expert_idx + 1) * expert_size];

    // Compare
    let mut max_diff: f32 = 0.0;
    for i in 0..expert_size {
        let diff = (gpu_result[i] - cpu_expert[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("GPU first 10: {:?}", &gpu_result[..10]);
    println!("CPU first 10: {:?}", &cpu_expert[..10]);
    println!("max diff: {}", max_diff);
    assert!(max_diff < 0.01, "GPU vs CPU mismatch: max_diff={}", max_diff);

    println!("✅ Real data dequant kernel matches CPU reference");
}



pub fn qwen35_test_expert_matmul(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let gpu = &mut gpus[0];

    let ptx = Ptx::from_file("ptx/mlx_dequant.ptx");
    gpu.dev.load_ptx(ptx, "mlx_dequant", &["mlx_dequant_4bit"]).unwrap();

    // Upload packed gate_proj
    let packed = load_tensor_u32(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight");
    let scales_f32 = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales");
    let biases_f32 = load_scales(&mut st, "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases");
    let d_packed = gpu.upload_u32(&packed, "packed");
    let d_scales = gpu.upload_f16_from_f32(&scales_f32, "scales");
    let d_biases = gpu.upload_f16_from_f32(&biases_f32, "biases");

    let expert_idx = 5usize;
    let out_dim = 1024;
    let in_dim = 3072;
    let packed_cols = in_dim / 8;
    let scale_cols = in_dim / 64;

    // Scratch buffer for one dequanted expert
    let d_scratch: CudaSlice<f16> = gpu.dev.alloc_zeros::<f16>(out_dim * in_dim).unwrap();

    // Dequant expert 5
    let func = gpu.dev.get_func("mlx_dequant", "mlx_dequant_4bit").unwrap();
    let total_packed = (out_dim * packed_cols) as u32;
    let block_size = 256u32;
    let grid_size = (total_packed + block_size - 1) / block_size;
    let packed_offset = expert_idx * out_dim * packed_cols;
    let scale_offset = expert_idx * out_dim * scale_cols;

    unsafe {
        func.launch(
            cudarc::driver::LaunchConfig {
                grid_dim: (grid_size, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &d_packed.slice(packed_offset..),
                &d_scales.slice(scale_offset..),
                &d_biases.slice(scale_offset..),
                &d_scratch,
                out_dim as i32,
                packed_cols as i32,
                scale_cols as i32,
                64i32,
            ),
        ).unwrap();
    }

    // Create a dummy input (e.g. all 0.01)
    let input = vec![0.01f32; in_dim];

    // GPU matmul: input [3072] x scratch [1024, 3072]^T -> [1024]
    let gpu_result = gpu.matmul_transposed_f16(&input, &d_scratch, in_dim, out_dim);

    // CPU reference: full dequant, slice expert 5, manual matmul
    let cpu_full = load_mlx_affine(&mut st,
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight",
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales",
                                   "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases",
                                   config.bits(), config.group_size());
    let expert_size = out_dim * in_dim;
    let cpu_expert = &cpu_full[expert_idx * expert_size..(expert_idx + 1) * expert_size];
    let mut cpu_result = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut dot = 0.0f32;
        for j in 0..in_dim {
            dot += input[j] * cpu_expert[i * in_dim + j];
        }
        cpu_result[i] = dot;
    }

    // Compare
    let mut max_diff: f32 = 0.0;
    for i in 0..out_dim {
        let diff = (gpu_result[i] - cpu_result[i]).abs();
        if diff > max_diff { max_diff = diff; }
    }
    println!("GPU matmul first 10: {:?}", &gpu_result[..10]);
    println!("CPU matmul first 10: {:?}", &cpu_result[..10]);
    println!("max diff: {}", max_diff);
    assert!(max_diff < 0.1, "Expert matmul mismatch: max_diff={}", max_diff);

    println!("✅ Expert dequant + matmul matches CPU reference");
}



pub fn qwen35_test_load_model_packed(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);

    let entry = &st.entries["language_model.model.layers.0.mlp.shared_expert_gate.weight"];
    println!("shared_expert_gate shape: {:?} dtype: {}", entry.shape, entry.dtype);    let model = Qwen35Model::new(&mut st, gpus);
    println!("Layers: {}", model.layers.len());

    for (i, gpu) in gpus.iter().enumerate() {
        let (free, total) = gpu.mem_info();
        println!("GPU {}: {:.1} GB free / {:.1} GB total",
                 i, free as f64 / 1024.0 / 1024.0 / 1024.0, total as f64 / 1024.0 / 1024.0 / 1024.0);
    }

    assert_eq!(model.layers.len(), 48);
    println!("✅ Full model loaded with packed experts");
}


pub fn qwen35_test_moe_forward(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    println!("Loading layer 0 MoE weights...");
    let prefix = "language_model.model.layers.0";
    let moe = load_moe_weights(&mut st, gpu, prefix, config.bits(), config.group_size(), 0, tc);

    // Dummy input
    let mut hidden = vec![0.01f32; tc.hidden_size];
    let normed = hidden.clone();

    println!("Running MoE forward...");
    qwen35_moe_ffn(&mut hidden, &normed, &moe, tc.num_experts_per_tok, gpu);

    let nonzero = hidden.iter().filter(|&&v| v != 0.01).count();
    let max_val = hidden.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("hidden changed: {}/{}", nonzero, tc.hidden_size);
    println!("max abs: {}", max_val);
    assert!(nonzero > 0, "MoE should modify hidden state");

    println!("✅ MoE forward works");
}

pub fn qwen35_test_qkv(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    let qkv = qwen35_compute_qkv(&normed, &layer, tc, gpu);

    println!("q len={}, expected={}", qkv.q.len(), tc.num_attention_heads * tc.head_dim);
    println!("k len={}, expected={}", qkv.k.len(), tc.num_key_value_heads * tc.head_dim);
    println!("v len={}, expected={}", qkv.v.len(), tc.num_key_value_heads * tc.head_dim);
    println!("gate len={}, expected={}", qkv.gate.len(), tc.num_attention_heads * tc.head_dim);

    assert_eq!(qkv.q.len(), tc.num_attention_heads * tc.head_dim);
    assert_eq!(qkv.k.len(), tc.num_key_value_heads * tc.head_dim);
    assert_eq!(qkv.v.len(), tc.num_key_value_heads * tc.head_dim);
    assert_eq!(qkv.gate.len(), tc.num_attention_heads * tc.head_dim);

    println!("q first 5: {:?}", &qkv.q[..5]);
    println!("k first 5: {:?}", &qkv.k[..5]);
    println!("gate first 5: {:?}", &qkv.gate[..5]);

    println!("✅ QKV projection + norms correct");
}

pub fn qwen35_test_rope(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
    let mut qkv = qwen35_compute_qkv(&normed, &layer, tc, gpu);

    let q_before = qkv.q[..5].to_vec();
    let nope_dim = tc.head_dim - (tc.head_dim as f32 * tc.rope_parameters.partial_rotary_factor) as usize;

    qwen35_apply_rope(&mut qkv.q, &mut qkv.k, tc, 0);

    // At position 0, cos=1 sin=0, so rope dims should be unchanged
    println!("q before rope: {:?}", &q_before);
    println!("q after rope:  {:?}", &qkv.q[..5]);

    // Nope dims (first 192) should be untouched
    println!("q nope unchanged: {}", q_before[0] == qkv.q[0]);

    // Test position > 0 to see rope actually changes values
    let mut qkv2 = qwen35_compute_qkv(&normed, &layer, tc, gpu);
    qwen35_apply_rope(&mut qkv2.q, &mut qkv2.k, tc, 5);

    // Nope dims should still be same as pre-rope
    println!("q nope at pos 5 unchanged: {}", q_before[0] == qkv2.q[0]);

    // Rope dims should differ between pos 0 and pos 5
    let rope_start = nope_dim;
    let rope_differs = qkv.q[rope_start] != qkv2.q[rope_start];
    println!("q rope dim differs pos0 vs pos5: {}", rope_differs);

    assert!(rope_differs, "RoPE should change values at different positions");
    println!("✅ Partial RoPE works");
}


pub fn qwen35_test_full_attn_forward(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);
    let mut hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    // QKV + norms
    let mut qkv = qwen35_compute_qkv(&normed, &layer, tc, gpu);

    // RoPE
    qwen35_apply_rope(&mut qkv.q, &mut qkv.k, tc, 0);

    // Cache
    kv_cache.append(3, qkv.k, qkv.v);

    // Attention
    let attn_out = qwen35_gqa_attention(&qkv.q, &kv_cache.k[3], &kv_cache.v[3], tc);
    println!("attn_out len={}, first 5: {:?}", attn_out.len(), &attn_out[..5]);

    // Gate + project
    let projected = qwen35_gate_and_project(&attn_out, &qkv.gate, &layer.o_proj_w, tc.hidden_size, gpu);
    println!("projected len={}, first 5: {:?}", projected.len(), &projected[..5]);

    add_to_vector(&mut hidden, &projected);
    println!("hidden first 5: {:?}", &hidden[..5]);

    let changed = hidden.iter().filter(|&&v| v != 0.01).count();
    println!("hidden changed: {}/{}", changed, tc.hidden_size);
    assert!(changed > 0);

    println!("✅ Full attention forward chain works");
}


pub fn qwen35_test_full_attention_fn(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);
    let mut hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    qwen35_full_attention(&mut hidden, &normed, &layer, tc, &mut kv_cache, 3, 0, gpu);

    println!("hidden first 5: {:?}", &hidden[..5]);
    let changed = hidden.iter().filter(|&&v| v != 0.01).count();
    println!("hidden changed: {}/{}", changed, tc.hidden_size);
    assert!(changed > 0);

    // Second token at position 1 — verifies KV cache works
    let normed2 = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
    qwen35_full_attention(&mut hidden, &normed2, &layer, tc, &mut kv_cache, 3, 1, gpu);

    println!("after pos 1, hidden first 5: {:?}", &hidden[..5]);
    assert_eq!(kv_cache.k[3].len(), 2, "KV cache should have 2 entries");

    println!("✅ qwen35_full_attention works");
}


pub fn qwen35_test_conv1d(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    // Project to qkv
    let qkv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim;
    let qkv_proj = matmul_transposed_f16(&normed, &layer.in_proj_qkv_w, qkv_dim, gpu);
    println!("qkv_proj len={}, expected={}", qkv_proj.len(), qkv_dim);

    // Conv1d
    let mut conv_state = Vec::new();
    let kernel_size = tc.linear_conv_kernel_dim;
    let qkv_after = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, kernel_size);

    println!("conv_state len={}", conv_state.len());
    println!("qkv_after first 10: {:?}", &qkv_after[..10]);

    // Second token — conv state should affect output
    let qkv_after2 = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, kernel_size);
    let differs = qkv_after[0] != qkv_after2[0];
    println!("second token differs: {}", differs);

    let diff = (qkv_after[0] - qkv_after2[0]).abs();
    println!("diff[0]: {}", diff);

    println!("conv weight ch0: {:?}", &layer.conv1d_w[0..4]);

    println!("conv_state ch0 after 1st: {:?}", &conv_state[0..4]);
    let qkv_after2 = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, kernel_size);
    println!("conv_state ch0 after 2nd: {:?}", &conv_state[0..4]);
    println!("qkv_after[0]:  {}", qkv_after[0]);
    println!("qkv_after2[0]: {}", qkv_after2[0]);

    println!("✅ Conv1d update works");
}


pub fn qwen35_test_conv1d_debug(gpus: &mut Vec<Gpu>) {
    // Synthetic test: 2 channels, kernel_size=4
    let kernel_size = 4usize;
    let qkv_dim = 2usize;

    let input = vec![1.0f32, 2.0];
    let weights = vec![
        0.1, 0.2, 0.3, 0.4,  // ch0 weights
        0.5, 0.6, 0.7, 0.8,  // ch1 weights
    ];

    let mut conv_state = Vec::new();

    // Call 1: state starts as zeros
    let out1 = qwen35_conv1d_update(&input, &mut conv_state, &weights, qkv_dim, kernel_size);
    println!("state after call 1: {:?}", &conv_state);
    println!("out1: {:?}", out1);
    // Expected state: [0, 0, 0, 1.0, 0, 0, 0, 2.0]
    // Expected out1: [0*0.1 + 0*0.2 + 0*0.3 + 1.0*0.4, 0*0.5 + 0*0.6 + 0*0.7 + 2.0*0.8] = [0.4, 1.6]

    // Call 2: same input
    let out2 = qwen35_conv1d_update(&input, &mut conv_state, &weights, qkv_dim, kernel_size);
    println!("state after call 2: {:?}", &conv_state);
    println!("out2: {:?}", out2);
    // Expected state: [0, 0, 1.0, 1.0, 0, 0, 2.0, 2.0]
    // Expected out2: [0*0.1 + 0*0.2 + 1.0*0.3 + 1.0*0.4, 0*0.5 + 0*0.6 + 2.0*0.7 + 2.0*0.8] = [0.7, 3.0]

    assert!((out1[0] - 0.4).abs() < 1e-6 && (out1[1] - 1.6).abs() < 1e-6, "call 1 wrong: {:?}", out1);
    assert!((out2[0] - 0.7).abs() < 1e-6 && (out2[1] - 3.0).abs() < 1e-6, "call 2 wrong: {:?}", out2);
    println!("✅ Conv1d synthetic test passed");
}


pub fn qwen35_test_split_qkv(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    let qkv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim;
    let qkv_proj = matmul_transposed_f16(&normed, &layer.in_proj_qkv_w, qkv_dim, gpu);

    let mut conv_state = Vec::new();
    let qkv_conv = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, tc.linear_conv_kernel_dim);

    let split = qwen35_split_qkv(&qkv_conv, tc);

    println!("q len={}, expected={}", split.q.len(), tc.linear_num_key_heads * tc.linear_key_head_dim);
    println!("k len={}, expected={}", split.k.len(), tc.linear_num_key_heads * tc.linear_key_head_dim);
    println!("v len={}, expected={}", split.v.len(), tc.linear_num_value_heads * tc.linear_value_head_dim);

    // Verify L2 norm: each head of q should have unit norm
    let head_dim = tc.linear_key_head_dim;
    for h in 0..tc.linear_num_key_heads {
        let offset = h * head_dim;
        let norm: f32 = split.q[offset..offset + head_dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "q head {} norm={}, expected ~1.0", h, norm);
    }

    println!("q first 5: {:?}", &split.q[..5]);
    println!("✅ Split QKV + L2 norm correct");
}

pub fn qwen35_test_delta_gates(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    let gates = qwen35_compute_delta_gates(&normed, &layer, tc, gpu);

    println!("decay len={}, first 5: {:?}", gates.decay.len(), &gates.decay[..5]);
    println!("beta len={}, first 5: {:?}", gates.beta.len(), &gates.beta[..5]);

    assert_eq!(gates.decay.len(), tc.linear_num_value_heads);
    assert_eq!(gates.beta.len(), tc.linear_num_value_heads);

    // Decay should be in (0, 1)
    for (i, &d) in gates.decay.iter().enumerate() {
        assert!(d > 0.0 && d <= 1.0, "decay[{}]={} out of range", i, d);
    }
    // Beta should be in (0, 1) (sigmoid output)
    for (i, &b) in gates.beta.iter().enumerate() {
        assert!(b > 0.0 && b < 1.0, "beta[{}]={} out of range", i, b);
    }

    println!("✅ Delta gates correct");
}

pub fn qwen35_test_recurrent(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    let qkv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2
        + tc.linear_num_value_heads * tc.linear_value_head_dim;
    let qkv_proj = matmul_transposed_f16(&normed, &layer.in_proj_qkv_w, qkv_dim, gpu);
    let mut conv_state = Vec::new();
    let qkv_conv = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, tc.linear_conv_kernel_dim);
    let qkv = qwen35_split_qkv(&qkv_conv, tc);
    let gates = qwen35_compute_delta_gates(&normed, &layer, tc, gpu);

    let mut recurrent_state = Vec::new();
    let output = qwen35_recurrent_update(&qkv, &gates, &mut recurrent_state, tc);

    println!("output len={}, expected={}", output.len(), tc.linear_num_value_heads * tc.linear_value_head_dim);
    println!("output first 5: {:?}", &output[..5]);
    let max_abs = output.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("max_abs: {}", max_abs);

    // Second token should differ
    let output2 = qwen35_recurrent_update(&qkv, &gates, &mut recurrent_state, tc);
    let diff = (output[0] - output2[0]).abs();
    println!("second token diff[0]: {}", diff);
    assert!(diff > 0.0, "Second token should differ");

    println!("✅ Recurrent update works");
}


pub fn qwen35_test_linear_attn_output(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    // Fake recurrent output
    let recurrent_out = vec![0.01f32; tc.linear_num_value_heads * tc.linear_value_head_dim];

    let projected = qwen35_linear_attn_output(&recurrent_out, &normed, &layer, tc, gpu);

    println!("projected len={}, expected={}", projected.len(), tc.hidden_size);
    println!("projected first 5: {:?}", &projected[..5]);
    assert_eq!(projected.len(), tc.hidden_size);

    println!("✅ Linear attention output works");
}

pub fn qwen35_test_linear_attention_fn(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);
    let mut hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    qwen35_linear_attention(&mut hidden, &normed, &layer, tc, &mut kv_cache, 0, gpu);

    let changed = hidden.iter().filter(|&&v| v != 0.01).count();
    println!("hidden changed: {}/{}", changed, tc.hidden_size);
    println!("hidden first 5: {:?}", &hidden[..5]);

    // Second token
    let normed2 = rms_norm(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
    let hidden_before = hidden[0];
    qwen35_linear_attention(&mut hidden, &normed2, &layer, tc, &mut kv_cache, 0, gpu);
    println!("after pos 1, hidden first 5: {:?}", &hidden[..5]);
    assert!(hidden[0] != hidden_before, "Second token should change hidden");

    println!("✅ qwen35_linear_attention works");
}

pub fn qwen35_test_single_token(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);

    println!("Loading model...");
    let model = Qwen35Model::new(&mut st, gpus);
    let tc = &model.config.text_config;

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);

    println!("Running single token...");
    let token_id = 2; // arbitrary token
    let output_id = qwen35_generate_token(token_id, 0, &model, &mut kv_cache, gpus);

    println!("input token: {}, output token: {}", token_id, output_id);
    assert!(output_id < tc.vocab_size, "output token should be valid");

    println!("✅ Single token forward works");
}


pub fn qwen35_test_layer0_debug(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    // Load just layer 0
    let layer = Qwen35LinearAttnLayer::load(&mut st, 0, gpu, &config);

    // Embed token 760
    let embed_weights = load_mlx_affine(&mut st,
                                        "language_model.model.embed_tokens.weight",
                                        "language_model.model.embed_tokens.scales",
                                        "language_model.model.embed_tokens.biases",
                                        config.bits(), config.group_size());
    let embed_start = 760 * tc.hidden_size;
    let embed = &embed_weights[embed_start..embed_start + tc.hidden_size];

    // Normed
    let normed = rms_norm_zero_centered(embed, &layer.input_norm_w, tc.rms_norm_eps);
    println!("Rust normed first 10: {:?}", &normed[..10]);

    // QKV projection
    let qkv_dim = tc.linear_num_key_heads * tc.linear_key_head_dim * 2
        + tc.linear_num_value_heads * tc.linear_value_head_dim;
    let qkv_proj = matmul_transposed_f16(&normed, &layer.in_proj_qkv_w, qkv_dim, gpu);
    println!("Rust qkv_proj first 10: {:?}", &qkv_proj[..10]);
    // Python: [2.934036, -1.535046, -1.032008, -0.69264, -0.298411, 0.770717, -1.090574, 0.403697, 1.010088, 0.530222]

    // Conv1d (first token, so state starts empty)
    let mut conv_state = Vec::new();
    let qkv_conv = qwen35_conv1d_update(&qkv_proj, &mut conv_state, &layer.conv1d_w, qkv_dim, tc.linear_conv_kernel_dim);
    let qkv_activated = silu(&qkv_conv);
    println!("Rust qkv after conv+silu first 10: {:?}", &qkv_activated[..10]);

    // Split + L2 norm
    let qkv = qwen35_split_qkv(&qkv_activated, tc);
    println!("Rust q first 5: {:?}", &qkv.q[..5]);
    println!("Rust k first 5: {:?}", &qkv.k[..5]);
    println!("Rust v first 5: {:?}", &qkv.v[..5]);

    // Gates
    let gates = qwen35_compute_delta_gates(&normed, &layer, tc, gpu);
    println!("Rust decay first 5: {:?}", &gates.decay[..5]);
    println!("Rust beta first 5: {:?}", &gates.beta[..5]);

    // Recurrent
    let mut recurrent_state = Vec::new();
    let recurrent_out = qwen35_recurrent_update(&qkv, &gates, &mut recurrent_state, tc);
    let s7_offset = 7 * 128 * 128;

    // Manual S @ q for head 7
    let s_offset = 7 * 128 * 128;
    let k_offset = 1 * 128; // k_h = 1
    let k_dim = 128;
    let v_dim = 128;
    let mut manual_out = vec![0.0f32; v_dim];
    for vi in 0..v_dim {
        for ki in 0..k_dim {
            manual_out[vi] += recurrent_state[s_offset + ki * v_dim + vi] * qkv.q[k_offset + ki];
        }
    }
    let manual_max = manual_out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));

    let recurrent_out = qwen35_recurrent_update(&qkv, &gates, &mut recurrent_state, tc);
    println!("AFTER return head 7 max: {}", recurrent_out[896..1024].iter().cloned().fold(0.0f32, |a, b| a.max(b.abs())));
    println!("AFTER return head 7 first 5: {:?}", &recurrent_out[896..901]);
}


pub fn qwen35_test_generate(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);

    let tokenizer = BpeTokenizer::load(path, "\u{0120}".to_string(), "\u{010A}".to_string(), false);

    println!("Loading model...");
    let model = Qwen35Model::new(&mut st, gpus);
    let tc = &model.config.text_config;


    for i in 0..8 {
        let kind = match &model.layers[i] {
            Qwen35Layer::Linear(_) => "linear",
            Qwen35Layer::Full(_) => "full",
        };
        println!("----layer {}: {}", i, kind);
    }

    let prompt = "The capital of France is";
    let input_ids = tokenizer.encode(prompt);
    println!("Input tokens: {:?}", input_ids);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);

    let mut last_token = 0;
    for (i, &token_id) in input_ids.iter().enumerate() {
        last_token = qwen35_generate_token(token_id as usize, i, &model, &mut kv_cache, gpus);
        println!("pos {} token {} -> {}", i, token_id, last_token);
    }

    let mut all_ids = input_ids.clone();
    all_ids.push(last_token as u32);

    for _ in 0..10 {
        let pos = all_ids.len() - 1;
        let token_id = *all_ids.last().unwrap() as usize;
        let next = qwen35_generate_token(token_id, pos, &model, &mut kv_cache, gpus);
        all_ids.push(next as u32);
    }

    let result = tokenizer.decode(&all_ids);
    println!("Qwen3.5: {}", result);

    if let Qwen35Layer::Linear(l) = &model.layers[0] {
        println!("linear norm_w first 5: {:?}", &l.norm_w[..5]);
    }
}



pub fn qwen35_test_full_attn_layer3(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;

    let embed_weights = load_mlx_affine(&mut st,
                                        "language_model.model.embed_tokens.weight",
                                        "language_model.model.embed_tokens.scales",
                                        "language_model.model.embed_tokens.biases",
                                        config.bits(), config.group_size());
    let embed_start = 760 * tc.hidden_size;
    let mut hidden = embed_weights[embed_start..embed_start + tc.hidden_size].to_vec();

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);

    for i in 0..3 {
        let gpu = &mut gpus[0];
        let layer = Qwen35LinearAttnLayer::load(&mut st, i, gpu, &config);
        let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
        qwen35_linear_attention(&mut hidden, &normed, &layer, tc, &mut kv_cache, i, gpu);
        let post_normed = rms_norm_zero_centered(&hidden, &layer.post_norm_w, tc.rms_norm_eps);
        qwen35_moe_ffn(&mut hidden, &post_normed, &layer.moe, tc.num_experts_per_tok, gpu);
    }

    println!("hidden before layer 3 first 5: {:?}", &hidden[..5]);

    let gpu = &mut gpus[0];
    let layer3 = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);
    let normed = rms_norm_zero_centered(&hidden, &layer3.input_norm_w, tc.rms_norm_eps);

    let attn_dim = tc.num_attention_heads * tc.head_dim;

    let q_gate_raw = matmul_transposed_f16(&normed, &layer3.q_proj_w, attn_dim * 2, gpu);

    let q_part = &q_gate_raw[..attn_dim];
    let q_max = q_part.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let q_mean = q_part.iter().sum::<f32>() / q_part.len() as f32;

    let g_part = &q_gate_raw[attn_dim..];
    let g_max = g_part.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    let g_mean = g_part.iter().sum::<f32>() / g_part.len() as f32;
    let sig_mean = g_part.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).sum::<f32>() / g_part.len() as f32;

    println!("Q part: max={:.4} mean={:.4}", q_max, q_mean);
    println!("Gate part: max={:.4} mean={:.4} sigmoid_mean={:.4}", g_max, g_mean, sig_mean);

    // Now try swapped split
    let g_part2 = &q_gate_raw[..attn_dim];
    let q_part2 = &q_gate_raw[attn_dim..];
    let sig_mean2 = g_part2.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).sum::<f32>() / g_part2.len() as f32;
    println!("SWAPPED - Gate sigmoid_mean={:.4}", sig_mean2);
}


pub fn qwen35_test_layer_by_layer(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;

    let embed_weights = load_mlx_affine(&mut st,
                                        "language_model.model.embed_tokens.weight",
                                        "language_model.model.embed_tokens.scales",
                                        "language_model.model.embed_tokens.biases",
                                        config.bits(), config.group_size());

    let embed_start = 760 * tc.hidden_size;
    let mut hidden = embed_weights[embed_start..embed_start + tc.hidden_size].to_vec();
    println!("embed first 5: {:?}", &hidden[..5]);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);

    for i in 0..4 {
        let gpu = &mut gpus[0];
        let is_linear = config.is_linear_attention(i);
        let prefix = if is_linear { "linear" } else { "full" };
        println!("\n=== Layer {} ({}) ===", i, prefix);
        println!("  hidden before first 5: {:?}", &hidden[..5]);

        if is_linear {
            let layer = Qwen35LinearAttnLayer::load(&mut st, i, gpu, &config);
            let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
            println!("  normed first 5: {:?}", &normed[..5]);
            qwen35_linear_attention(&mut hidden, &normed, &layer, tc, &mut kv_cache, i, gpu);
            println!("  hidden after attn first 5: {:?}", &hidden[..5]);
            let post_normed = rms_norm_zero_centered(&hidden, &layer.post_norm_w, tc.rms_norm_eps);
            qwen35_moe_ffn(&mut hidden, &post_normed, &layer.moe, tc.num_experts_per_tok, gpu);
            println!("  hidden after MoE first 5: {:?}", &hidden[..5]);
        } else {
            let layer = Qwen35FullAttnLayer::load(&mut st, i, gpu, &config);
            let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
            println!("  normed first 5: {:?}", &normed[..5]);
            qwen35_full_attention(&mut hidden, &normed, &layer, tc, &mut kv_cache, i, 0, gpu);
            println!("  hidden after attn first 5: {:?}", &hidden[..5]);
            let post_normed = rms_norm_zero_centered(&hidden, &layer.post_norm_w, tc.rms_norm_eps);
            qwen35_moe_ffn(&mut hidden, &post_normed, &layer.moe, tc.num_experts_per_tok, gpu);
            println!("  hidden after MoE first 5: {:?}", &hidden[..5]);
        }
    }

    println!("\nFinal hidden first 10: {:?}", &hidden[..10]);
    let max_val = hidden.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("Final hidden max: {}", max_val);
}

pub fn qwen35_test_full(gpus: &mut Vec<Gpu>) {
    let path = PATH;
    let mut st = SafeTensor::new(path);
    let tokenizer = BpeTokenizer::load(path, "\u{0120}".to_string(), "\u{010A}".to_string(), false);

    println!("Loading model...");
    let model = Qwen35Model::new(&mut st, gpus);
    let tc = &model.config.text_config;

    // --- Part 1: Layer-by-layer debug for token 760 ---
    println!("\n=== LAYER-BY-LAYER DEBUG ===");
    let embed_start = 760 * tc.hidden_size;
    let mut hidden = model.embed_weights[embed_start..embed_start + tc.hidden_size].to_vec();
    println!("embed first 5: {:?}", &hidden[..5]);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);

    for i in 0..4 {
        let gpu = &mut gpus[0];
        match &model.layers[i] {
            Qwen35Layer::Linear(layer) => {
                println!("\n--- Layer {} (linear) ---", i);
                println!("  hidden before first 5: {:?}", &hidden[..5]);
                let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
                println!("  normed first 5: {:?}", &normed[..5]);
                qwen35_linear_attention(&mut hidden, &normed, layer, tc, &mut kv_cache, i, gpu);
                println!("  hidden after attn first 5: {:?}", &hidden[..5]);
                let post_normed = rms_norm_zero_centered(&hidden, &layer.post_norm_w, tc.rms_norm_eps);
                qwen35_moe_ffn(&mut hidden, &post_normed, &layer.moe, tc.num_experts_per_tok, gpu);
                println!("  hidden after MoE first 5: {:?}", &hidden[..5]);
            }
            Qwen35Layer::Full(layer) => {
                println!("\n--- Layer {} (full) ---", i);
                println!("  hidden before first 5: {:?}", &hidden[..5]);
                let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
                println!("  normed first 5: {:?}", &normed[..5]);
                qwen35_full_attention(&mut hidden, &normed, layer, tc, &mut kv_cache, i, 0, gpu);
                println!("  hidden after attn first 5: {:?}", &hidden[..5]);
                let post_normed = rms_norm_zero_centered(&hidden, &layer.post_norm_w, tc.rms_norm_eps);
                qwen35_moe_ffn(&mut hidden, &post_normed, &layer.moe, tc.num_experts_per_tok, gpu);
                println!("  hidden after MoE first 5: {:?}", &hidden[..5]);
                println!("  input_norm_w[0]={}, q_norm_w[0]={}", layer.input_norm_w[0], layer.q_norm_w[0]);
            }
        }
    }

    // --- Part 2: Full generation ---
    println!("\n=== GENERATION ===");
    let prompt = "The capital of France is";
    let input_ids = tokenizer.encode(prompt);
    println!("Input tokens: {:?}", input_ids);

    let mut kv_cache2 = KVCache::new(tc.num_hidden_layers);
    let mut last_token = 0;
    for (i, &token_id) in input_ids.iter().enumerate() {
        last_token = qwen35_generate_token(token_id as usize, i, &model, &mut kv_cache2, gpus);
        println!("pos {} token {} -> {}", i, token_id, last_token);
    }

    let mut all_ids = input_ids.clone();
    all_ids.push(last_token as u32);

    for _ in 0..20 {
        let pos = all_ids.len() - 1;
        let token_id = *all_ids.last().unwrap() as usize;
        let next = qwen35_generate_token(token_id, pos, &model, &mut kv_cache2, gpus);
        all_ids.push(next as u32);
    }

    let result = tokenizer.decode(&all_ids);
    println!("Qwen3.5: {}", result);
}


pub fn qwen35_affine_tests(gpus: &mut Vec<Gpu>, max_tokens: usize) {
    let path= PATH;

    let qwen_tokenizer = BpeTokenizer::load(
        &path,
        "\u{0120}".to_string(),
        "\u{010A}".to_string(),
        false
    );

    let mut safe_tensor_qwen = SafeTensor::new(&path);
    let qwen_model = LlamaModel::new(&mut safe_tensor_qwen, gpus);

    let prompt = "The capital of France is";

    let result = generate(GenerateArgs {
        prompt,
        tokenizer: &qwen_tokenizer,
        max_tokens,
        gpus,
        eos_token_id: &qwen_model.config.eos_token_id,
        num_layers: qwen_model.config.num_hidden_layers,
    }, &qwen_model);

    println!("Qwen: {}", result);

    // qwen_tests
    let expected = "The capital of France is Paris. The capital of Spain is Madrid. The capital of Italy is Rome. The capital of Germany is Berlin. The capital of the Netherlands is Amsterdam. The capital of Belgium is Brussels. The capital of Luxembourg is Luxembourg City. The capital of Austria is";
    assert_eq!(result.trim(), expected.trim(), "Qwen output mismatch");
}

pub fn qwen35_test_attn_dummy(gpus: &mut Vec<Gpu>) {
    let path = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit";
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);
    println!("normed first 5: {:?}", &normed[..5]);

    let mut kv_cache = KVCache::new(tc.num_hidden_layers);
    let mut hidden_mut = hidden.clone();
    qwen35_full_attention(&mut hidden_mut, &normed, &layer, tc, &mut kv_cache, 3, 0, gpu);

    let attn_out: Vec<f32> = hidden_mut.iter().zip(hidden.iter()).map(|(a, b)| a - b).collect();
    println!("attn_out first 10: {:?}", &attn_out[..10]);
    println!("attn_out max: {}", attn_out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs())));
    println!("result first 10: {:?}", &hidden_mut[..10]);
}


pub fn qwen35_test_attn_detail(gpus: &mut Vec<Gpu>) {
    let path = "/mnt/data/models/Qwen3.5-122B-A10B-AWQ-4bit";
    let mut st = SafeTensor::new(path);
    let config = Qwen35Config::load(&format!("{}/config.json", path));
    let tc = &config.text_config;
    let gpu = &mut gpus[0];

    let layer = Qwen35FullAttnLayer::load(&mut st, 3, gpu, &config);
    println!("q_norm_w first 5: {:?}", &layer.q_norm_w[..5]);

    let hidden = vec![0.01f32; tc.hidden_size];
    let normed = rms_norm_zero_centered(&hidden, &layer.input_norm_w, tc.rms_norm_eps);

    // Step by step
    let mut qkv = qwen35_compute_qkv(&normed, &layer, tc, gpu);

    qwen35_apply_rope(&mut qkv.q, &mut qkv.k, tc, 0);

    // Single token: attn_out = v (with GQA)
    let mut kv_cache = KVCache::new(tc.num_hidden_layers);
    kv_cache.append(3, qkv.k.clone(), qkv.v.clone());
    let attn_out = qwen35_gqa_attention(&qkv.q, &kv_cache.k[3], &kv_cache.v[3], tc);

    let projected = qwen35_gate_and_project(&attn_out, &qkv.gate, &layer.o_proj_w, tc.hidden_size, gpu);
}