use std::sync::Arc;
use cudarc::cublas::CudaBlas;
use cudarc::cublas::result::sgemm;
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut};
use cudarc::driver::result::mem_get_info;
use crate::common_types::{Arrf32, Vecf32};

pub type CudaArrf32 = CudaSlice<f32>;

pub struct Gpu {
    pub gpu_index: usize,
    pub dev: Arc<CudaDevice>,
    pub blas: CudaBlas,
}

pub fn get_gpu(gpus: &Vec<Gpu>, layer_no: usize, num_layers: usize) -> &Gpu {
    let gpu_idx = layer_no * gpus.len() / num_layers;

    &gpus[gpu_idx]
}


pub fn get_default_gpu<'a>(gpus: &'a Vec<Gpu>, _purpose: &str) -> &'a Gpu {
    &gpus[0]
}

impl Gpu {

    pub fn multiple(num_gpus: usize) -> Vec<Gpu> {
        let gpus = (0..num_gpus).map(|x| Gpu::new(x)).collect::<Vec<_>>();

        gpus
    }

    pub fn new(gpu_index: usize) -> Self {
        let dev = CudaDevice::new(gpu_index).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        Gpu { dev, blas, gpu_index }
    }

    pub fn mem_info(&self) -> (usize, usize) {
        let (free, total) = mem_get_info().unwrap();
        (free, total)
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

    pub fn upload(&self, data: &Arrf32, weight_name: &str) -> CudaSlice<f32> {
        //let (free, total) = self.mem_info();
        //let size_mb = (data.len() * 4) as f64 / 1024.0 / 1024.0;
        //let free_mb = free as f64 / 1024.0 / 1024.0;
        //println!("GPU {} upload: {:60} {:8.1} MB  (free: {:.0}/{:.0} MB)", self.gpu_index, weight_name, size_mb, free_mb, total as f64 / 1024.0 / 1024.0);
        self.dev.htod_sync_copy(data).unwrap()
    }
}