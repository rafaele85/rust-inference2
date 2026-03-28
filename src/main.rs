mod matmul;
mod activation;
mod kv_cache;
mod common_types;
mod gpu;
mod arch;
mod linear;
mod tokenizer;
mod generate;

mod tests;

use crate::gpu::Gpu;
use crate::tests::gpt2_tests::gpt2_tests;
use crate::tests::llama_tests::llama_tests;
use crate::tests::qwen_tests::{awq_matmul_test, qwen3_moe_tests, qwen_awq_e2e_test, qwen_awq_tests, qwen_tests};

fn main() {
    let gpus = &mut Gpu::multiple(3);

    // qwen_awq_e2e_test(gpus);  //6 s
    // qwen3_moe_tests(gpus);  //34 s

    // qwen_tests(gpus);  //1.5 min
    // llama_tests(gpus);  //7s
    // gpt2_tests(gpus);  //2s

}
