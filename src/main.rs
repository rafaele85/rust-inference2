use crate::root::gpu::Gpu;
use crate::root::tests::qwen_tests::qwen_awq_e2e_test;

mod root;

fn main() {
    let gpus = &mut Gpu::multiple(3);

    qwen_awq_e2e_test(gpus, 50);  //6 s
    // qwen3_moe_tests(gpus);  //34 s

    // qwen_tests(gpus);  //1.5 min
    // llama_tests(gpus);  //7s
    // gpt2_tests(gpus);  //2s

}
