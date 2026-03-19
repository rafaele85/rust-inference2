pub fn extract_q(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[0..dim].to_vec()
}

pub fn extract_k(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[dim..dim * 2].to_vec()
}

pub fn extract_v(qkv: &[f32]) -> Vec<f32> {
    let dim = qkv.len() / 3;
    qkv[dim * 2..dim * 3].to_vec()
}

