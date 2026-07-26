use std::sync::Arc;
use cudarc::cublas::CudaBlas;
use cudarc::cublas::result::{hgemm, sgemm};
use cudarc::cublas::sys::cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync};
use cudarc::driver::result::mem_get_info;
use cudarc::nvrtc::Ptx;
use half::f16;
use crate::root::common_types::{Arrf16, Arrf32, Vecf16, Vecf32, WeightF16};

pub type CudaArrf32 = CudaSlice<f32>;
pub type CudaArrf16 = CudaSlice<f16>;

pub type CudaArru32 = CudaSlice<u32>;

fn f16_from_f32_arr(data: &Arrf32) -> Vecf16 {
    let res = data.iter().map(|x| f16::from_f32(*x)).collect::<Vec<_>>();

    res
}

pub struct GpuScratch {
    input_f32: CudaSlice<f32>,
    output_f32: CudaSlice<f32>,
    input_f16: CudaSlice<f16>,
    output_f16: CudaSlice<f16>,
    pub(crate) expert_f16: CudaSlice<f16>,
}

impl GpuScratch {
    fn new(dev: &Arc<CudaDevice>) -> Self {
        let init = 1;
        Self {
            input_f32: dev.alloc_zeros::<f32>(init).unwrap(),
            output_f32: dev.alloc_zeros::<f32>(init).unwrap(),
            input_f16: dev.alloc_zeros::<f16>(init).unwrap(),
            output_f16: dev.alloc_zeros::<f16>(init).unwrap(),
            expert_f16: dev.alloc_zeros::<f16>(1).unwrap(),
        }
    }

    /// Only safe for use with GEMM calls where beta = 0.0, which overwrites
    /// the entire output buffer. Do not reuse for other purposes without zeroing first.
    fn ensure_f32(&mut self, dev: &Arc<CudaDevice>, in_len: usize, out_len: usize) {
        if self.input_f32.len() != in_len {
            self.input_f32 = dev.alloc_zeros::<f32>(in_len).unwrap();
        }
        if self.output_f32.len() != out_len {
            self.output_f32 = dev.alloc_zeros::<f32>(out_len).unwrap();
        }
    }

    /// Only safe for use with GEMM calls where beta = 0.0, which overwrites
    /// the entire output buffer. Do not reuse for other purposes without zeroing first.
    fn ensure_f16(&mut self, dev: &Arc<CudaDevice>, in_len: usize, out_len: usize) {
        if self.input_f16.len() != in_len {
            self.input_f16 = dev.alloc_zeros::<f16>(in_len).unwrap();
        }
        if self.output_f16.len() != out_len {
            self.output_f16 = dev.alloc_zeros::<f16>(out_len).unwrap();
        }
    }

    pub(crate) fn ensure_expert_f16(&mut self, dev: &Arc<CudaDevice>, len: usize) {
        if self.expert_f16.len() != len {
            self.expert_f16 = dev.alloc_zeros::<f16>(len).unwrap();
        }
    }
}

pub struct Gpu {
    pub gpu_index: usize,
    pub dev: Arc<CudaDevice>,
    pub blas: CudaBlas,
    pub(crate) scratch: GpuScratch,
}

pub fn get_gpu(gpus: &mut Vec<Gpu>, layer_no: usize, num_layers: usize) -> &mut Gpu {
    //return get_gpu_weighted(gpus, layer_no, num_layers);
    let gpu_idx = layer_no * gpus.len() / num_layers;

    &mut gpus[gpu_idx]
}

pub fn get_gpu_weighted(gpus: &mut Vec<Gpu>, layer_no: usize, num_layers: usize) -> &mut Gpu {
    // Weights: 24, 32, 24 = total 80
    // GPU 0: 0-29% of layers, GPU 1: 30-69%, GPU 2: 70-100%
    let capacities = [24.0, 32.0, 24.0];
    let total: f64 = capacities.iter().sum();
    let progress = layer_no as f64 / num_layers as f64;

    let mut cumulative = 0.0;
    for (i, &cap) in capacities.iter().enumerate() {
        cumulative += cap / total;
        if progress < cumulative {
            return &mut gpus[i];
        }
    }
    let last = gpus.len() - 1;
    &mut gpus[last]
}


pub fn get_default_gpu<'a>(gpus: &'a mut Vec<Gpu>, _purpose: &str) -> &'a mut Gpu {
    &mut gpus[0]
}

impl Gpu {

    pub fn new(gpu_index: usize) -> Self {
        let dev = CudaDevice::new(gpu_index).unwrap();
        let blas = CudaBlas::new(dev.clone()).unwrap();
        let scratch = GpuScratch::new(&dev);

        let ptx = Ptx::from_file("ptx/ct_dequant_4bit.ptx");
        dev.load_ptx(ptx, "ct_dequant", &["ct_dequant_4bit"]).unwrap();

        Gpu { dev, blas, gpu_index, scratch }
    }

    pub fn multiple(num_gpus: usize) -> Vec<Gpu> {
        let gpus = (0..num_gpus).map(|x| Gpu::new(x)).collect::<Vec<_>>();

        gpus
    }


    pub fn mem_info(&self) -> (usize, usize) {
        let (free, total) = mem_get_info().unwrap();
        (free, total)
    }

    fn gpu_sgemm(&mut self, input: &Arrf32, d_weight: &CudaArrf32, in_dim: usize, out_dim: usize, transpose: bool) -> Vecf32 {
        self.scratch.ensure_f32(&self.dev, in_dim, out_dim);
        self.dev.htod_sync_copy_into(&input, &mut self.scratch.input_f32).unwrap();
        //self.dev.htod_sync_copy_into(&vec![0.0f32; out_dim], &mut self.scratch.output_f32).unwrap();

        let handle = *self.blas.handle();
        let trans = if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let transb = CUBLAS_OP_N;
        let m = out_dim as i32;
        let n = 1;
        let k = in_dim as i32;
        let alpha = &1.0f32 as *const f32;
        let a = *d_weight.device_ptr() as *const f32;
        let lda = if transpose { in_dim as i32 } else { out_dim as i32 };
        let b = *self.scratch.input_f32.device_ptr() as *const f32;
        let ldb = in_dim as i32;
        let beta = &0.0f32 as *const f32;
        let c = *self.scratch.output_f32.device_ptr_mut() as *mut f32;
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
        let result = self.dev.dtoh_sync_copy(&self.scratch.output_f32).unwrap();
        result
    }

    pub fn matmul(&mut self, input: &Arrf32, weight: &CudaArrf32, in_dim: usize, out_dim: usize) -> Vecf32 {
        self.gpu_sgemm(input, weight, in_dim, out_dim, false)
    }

    pub fn matmul_transposed(&mut self, input: &Arrf32, weight: &CudaArrf32, in_dim: usize, out_dim: usize) -> Vecf32 {
        self.gpu_sgemm(input, weight, in_dim, out_dim, true)
    }

    pub fn upload(&mut self, data: &Arrf32, _weight_name: &str) -> CudaSlice<f32> {
        //let (free, total) = self.mem_info();
        //let size_mb = (data.len() * 4) as f64 / 1024.0 / 1024.0;
        //let free_mb = free as f64 / 1024.0 / 1024.0;
        //println!("GPU {} upload: {:60} {:8.1} MB  (free: {:.0}/{:.0} MB)", self.gpu_index, weight_name, size_mb, free_mb, total as f64 / 1024.0 / 1024.0);
        self.dev.htod_sync_copy(data).unwrap()
    }

    pub fn upload_f16_from_f32(&mut self, data: &Arrf32, _weight_name: &str) -> CudaSlice<f16> {
        let f16_data = f16_from_f32_arr(data);
        self.dev.htod_sync_copy(&f16_data).unwrap()
    }

    pub fn upload_f16_from_bf16(&mut self, bf16_bytes: &[u8], _weight_name: &str) -> WeightF16 {
        let f16_data: Vecf16 = bf16_bytes.chunks_exact(2).map(|c| {
            let b = c.try_into().unwrap();
            let bits = u16::from_le_bytes(b);
            let f32_val = f32::from_bits((bits as u32) << 16);
            f16::from_f32(f32_val)
        }).collect();

        let res = self.dev.htod_sync_copy(&f16_data).unwrap();

        res
    }


    fn gpu_hgemm_f16(&mut self, input_f16: &Arrf16, d_weight: &CudaArrf16, in_dim: usize, out_dim: usize, transpose: bool) -> Vecf16 {
        self.scratch.ensure_f16(&self.dev, in_dim, out_dim);
        self.dev.htod_sync_copy_into(input_f16, &mut self.scratch.input_f16).unwrap();

        let handle = *self.blas.handle();
        let trans = if transpose { CUBLAS_OP_T } else { CUBLAS_OP_N };
        let transb = CUBLAS_OP_N;
        let m = out_dim as i32;
        let n = 1;
        let k = in_dim as i32;
        let alpha = f16::from_f32(1.0);
        let a = *d_weight.device_ptr() as *const f16;
        let lda = if transpose { in_dim as i32 } else { out_dim as i32 };
        let b = *self.scratch.input_f16.device_ptr() as *const f16;
        let ldb = in_dim as i32;
        let beta = f16::from_f32(0.0);
        let c = *self.scratch.output_f16.device_ptr_mut() as *mut f16;
        let ldc = out_dim as i32;
        unsafe {
            hgemm(
                handle,
                trans, transb,
                m, n, k,
                &alpha as *const f16,
                a, lda,
                b, ldb,
                &beta as *const f16,
                c, ldc
            ).unwrap();
        }
        let result_f16 = self.dev.dtoh_sync_copy(&self.scratch.output_f16).unwrap();
        result_f16
    }

    pub fn matmul_transposed_f16(&mut self, input: &Arrf32, weight: &CudaArrf16, in_dim: usize, out_dim: usize) -> Vecf32 {
        let input_f16: Vecf16 = input.iter().map(|x| f16::from_f32(*x)).collect();
        let output_f16 = self.gpu_hgemm_f16(&input_f16, weight, in_dim, out_dim, true);
        let output = output_f16.iter().map(|&x| x.to_f32()).collect::<Vec<_>>();

        output
    }

    pub fn upload_u32(&mut self, data: &[u32], _weight_name: &str) -> CudaSlice<u32> {
        self.dev.htod_sync_copy(data).unwrap()
    }
}
