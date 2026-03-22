use std::sync::Arc;
use cudarc::cublas::CudaBlas;
use cudarc::cublas::result::sgemm;
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use crate::common_types::{Arrf32, Vecf32};

pub type CudaArrf32 = CudaSlice<f32>;

pub struct Gpu {
    pub dev: Arc<CudaDevice>,
    pub blas: CudaBlas,
}

impl Gpu {
    pub fn new(gpu_index: usize) -> Self {
        let dev = CudaDevice::new(gpu_index).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        Gpu { dev, blas }
    }

    fn sgemm(&self, input: &Arrf32, d_weight: &CudaArrf32, in_dim: usize, out_dim: usize, transpose: bool) -> Vecf32 {
        let d_input = self.dev.htod_sync_copy(input).unwrap();
        let mut d_output = self.dev.alloc_zeros::<f32>(out_dim).unwrap();

        let handle = *self.blas.handle();
        let trans = if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let transb = CUBLAS_OP_N;
        let m = out_dim as i32;
        let n = 1;
        let k = in_dim as i32;
        let alpha = &1.0f32 as *const f32;
        let a = *d_weight.device_ptr() as *const f32;
        let lda = if transpose { in_dim as i32 } else { out_dim as i32 };
        let b = *d_input.device_ptr() as *const f32;
        let ldb = in_dim as i32;
        let beta = &0.0f32 as *const f32;
        let c = *d_output.device_ptr_mut() as *mut f32;
        let ldc = out_dim as i32;
        unsafe {
            sgemm(
                handle,
                trans, transb,
                m, n, k,
                alpha,
                a, lda, b, ldb,
                beta,
                c, ldc
            ).unwrap();
        }
        let result = self.dev.dtoh_sync_copy(&d_output).unwrap();
        result
    }

    pub fn matmul(&self, input: &Arrf32, weight: &CudaArrf32, in_dim: usize, out_dim: usize) -> Vec<f32> {
        self.sgemm(input, weight, in_dim, out_dim, false)
    }

    pub fn matmul_transposed(&self, input: &Arrf32, weight: &CudaArrf32, in_dim: usize, out_dim: usize) -> Vec<f32> {
        self.sgemm(input, weight, in_dim, out_dim, true)
    }

    pub fn upload(&self, data: &Arrf32) -> CudaSlice<f32> {
        self.dev.htod_sync_copy(data).unwrap()
    }
}