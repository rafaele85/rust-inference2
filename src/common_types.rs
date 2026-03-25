use crate::gpu::{CudaArrf32, Gpu};
use crate::kv_cache::KVCache;

pub type Vecf32 = Vec<f32>;

pub type Arrf32 = [f32];

pub type Weight = CudaArrf32;