use crate::root::gpu::Gpu;
use crate::root::tests::deepseek_tests::deepseek_tests;
use crate::root::tests::glm_tests::{glm_tests};
use crate::root::tests::gpt2_tests::gpt2_tests;
use crate::root::tests::llama_tests::llama_tests;
use crate::root::tests::qwen35_tests::{qwen35_test_attn_detail, qwen35_test_attn_dummy, qwen35_test_generate};

mod root;

fn main() {
    let gpus = &mut Gpu::multiple(3);


    //qwen_awq_e2e_test(gpus, 50);  //6 s
    // qwen3_moe_tests(gpus, 50);  //34 s
     // qwen_tests(gpus, 50);  //1.5 min
     // llama_tests(gpus, 50);  //7s
      // gpt2_tests(gpus, 50);  //2s
      //deepseek_tests(gpus, 50);
      // glm_tests(gpus, 50); //1:41 - works but incorrect space symbols
     // qwen_affine_tests(gpus, 50);  // Qwen3.5-122B-A10B-q4-affine-g64-mlx" - fails


    qwen35_test_attn_detail(gpus);
}
