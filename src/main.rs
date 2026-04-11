use crate::root::gpu::Gpu;
use crate::root::tests::deepseek_tests::deepseek_tests;
use crate::root::tests::glm_tests::{glm_tests};
use crate::root::tests::gpt2_tests::gpt2_tests;
use crate::root::tests::llama_tests::llama_tests;
use crate::root::tests::qwen_tests::{qwen35_test_config, qwen35_test_conv1d, qwen35_test_conv1d_debug, qwen35_test_delta_gates, qwen35_test_dequant, qwen35_test_dequant_kernel, qwen35_test_dequant_kernel_real, qwen35_test_expert_matmul, qwen35_test_expert_values, qwen35_test_full_attention_fn, qwen35_test_full_attn_forward, qwen35_test_fused_experts, qwen35_test_generate, qwen35_test_linear_attention_fn, qwen35_test_linear_attn_output, qwen35_test_load_both_layers, qwen35_test_load_full_attn, qwen35_test_load_full_attn2, qwen35_test_load_layer, qwen35_test_load_linear_attn, qwen35_test_load_model, qwen35_test_load_model_packed, qwen35_test_moe_forward, qwen35_test_qkv, qwen35_test_recurrent, qwen35_test_rope, qwen35_test_single_token, qwen35_test_split_qkv, qwen35_test_upload_packed, qwen35_test_upload_scales, qwen35_test_weights, qwen3_moe_tests, qwen_affine_tests, qwen_awq_e2e_test, qwen_tests};

mod root;

fn main() {
    let gpus = &mut Gpu::multiple(3);


    //qwen_awq_e2e_test(gpus, 50);  //6 s
    // qwen3_moe_tests(gpus, 50);  //34 s
     // qwen_tests(gpus, 50);  //1.5 min
     // llama_tests(gpus, 50);  //7s
      // gpt2_tests(gpus, 50);  //2s
     // deepseek_tests(gpus, 50);
     // glm_tests(gpus, 50); //1:41
    //qwen_affine_tests(gpus, 50);
    qwen35_test_generate(gpus);
}
