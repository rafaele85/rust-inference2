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
use crate::tests::qwen_tests::{qwen3_tests, qwen_tests};

fn main() {
    let gpus = &Gpu::multiple(3);

    qwen3_tests(gpus);

    qwen_tests(gpus);

    llama_tests(gpus);

    gpt2_tests(gpus);

}
