use half::f16;
use crate::root::gpu::{CudaArrf16, CudaArrf32};

pub type Vecf32 = Vec<f32>;

pub type Arrf32 = [f32];

pub type Vecf16 = Vec<f16>;

pub type Arrf16 = [f16];

pub type Arri32 = [i32];
pub type Veci32 = Vec<i32>;


pub type Weight = CudaArrf32;
pub type WeightF16 = CudaArrf16;