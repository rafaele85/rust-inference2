use crate::root::common_types::{Vecf32, WeightF16};
use crate::root::generate::GenerateNextTokenArgs;

pub trait GenerateNextToken {
    fn generate_next_token(&self, args: GenerateNextTokenArgs) -> usize;
}
pub trait Model {
    fn get_embed_weights(&self) -> &Vecf32;
    fn get_final_norm_w(&self) -> &Vecf32;
    fn get_lm_head_w(&self) -> &WeightF16;
}

