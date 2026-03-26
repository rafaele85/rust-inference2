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
use crate::tests::qwen_tests::{qwen3_moe_tests};

fn main() {
    let gpus = &Gpu::multiple(3);

    qwen3_moe_tests(gpus);

//    qwen_tests(gpus);
//    llama_tests(gpus);
//    gpt2_tests(gpus);

}
