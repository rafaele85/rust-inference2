use crate::root::gpu::Gpu;
use crate::root::tests::deepseek_tests::{deepseek_multi_token_test, deepseek_single_token_test, deepseek_tests};
use crate::root::tests::glm_tests::glm_tests;
use crate::root::tests::gpt2_tests::gpt2_tests;
use crate::root::tests::llama_tests::llama_tests;
use crate::root::tests::qwen_tests::{qwen3_moe_tests, qwen_awq_e2e_test, qwen_tests};

mod root;

fn main() {
    let gpus = &mut Gpu::multiple(3);


    // qwen_awq_e2e_test(gpus, 50);  //6 s
    // qwen3_moe_tests(gpus, 50);  //34 s

    // qwen_tests(gpus, 50);  //1.5 min
     // llama_tests(gpus, 50);  //7s
    // gpt2_tests(gpus, 50);  //2s
    //deepseek_tests(gpus, 50);
    glm_tests();
}
