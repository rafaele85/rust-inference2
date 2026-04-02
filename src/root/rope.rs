use crate::root::common_types::{Arrf32, Vecf32};

pub fn compute_rope_freqs(half: usize, pos: usize, theta: f32) -> (Vecf32, Vecf32) {
    let mut cos_vals = vec![0.0f32; half];
    let mut sin_vals = vec![0.0f32; half];

    for i in 0..half {
        let freq = 1.0 / theta.powf(i as f32 / half as f32);
        let angle = pos as f32 * freq;
        cos_vals[i] = angle.cos();
        sin_vals[i] = angle.sin();
    }

    (cos_vals, sin_vals)
}


pub struct ApplyRopeArgs<'a> {
    pub out: &'a mut Vecf32,
    pub x: &'a Arrf32,
    pub num_heads: usize,
    pub cos: &'a Arrf32,
    pub sin: &'a Arrf32,
    pub head_dim: usize,
    pub nope_dim: usize,
    pub interleaved: bool,
}

pub fn apply_rope<'a>(args: ApplyRopeArgs<'a>) -> &'a mut Vecf32 {
    let ApplyRopeArgs {out, x, num_heads, cos, sin, head_dim, nope_dim, interleaved} = args;
    for h in 0..num_heads {
        let offset = h * head_dim + nope_dim;
        apply_rope_to_head(ApplyRopeToHeadArgs {out, x, offset, cos, sin, interleaved});
    }

    out
}

pub struct ApplyRopeToHeadArgs<'a> {
    pub out: &'a mut Vecf32,
    pub x: &'a Arrf32,
    pub cos: &'a Arrf32,
    pub sin: &'a Arrf32,
    pub offset: usize,
    pub interleaved: bool,
}

pub fn apply_rope_to_head(args: ApplyRopeToHeadArgs) {
    let ApplyRopeToHeadArgs {x, offset, cos, sin, out, interleaved} = args;
    let half = cos.len();
    for i in 0..half {
        let (i1, i2) = if interleaved { (i * 2, i * 2 + 1) } else { (i, i + half) };
        let x1 = x[offset + i1];
        let x2 = x[offset + i2];
        out[offset + i1] = x1 * cos[i] - x2 * sin[i];
        out[offset + i2] = x2 * cos[i] + x1 * sin[i];
    }
}