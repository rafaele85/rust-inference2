
pub type KVVector = Vec<f32>;
pub type PositionKVVectors = Vec<KVVector>;
pub type LayerKVVectors = Vec<PositionKVVectors>;

pub struct KVCache {
    pub k: LayerKVVectors,
    pub v: LayerKVVectors,
    // GatedDeltaNet state (only used by linear attention layers)
    pub conv_state: PositionKVVectors,     // [num_layers][qkv_dim * kernel_size]
    pub recurrent_state: PositionKVVectors, // [num_layers][num_v_heads * key_head_dim * value_head_dim]
}
impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        let k = vec![Vec::new(); num_layers];
        let v = vec![Vec::new(); num_layers];
        let conv_state = vec![Vec::new(); num_layers];
        let recurrent_state = vec![Vec::new(); num_layers];

        KVCache { k, v, conv_state, recurrent_state }
    }

    pub fn append(&mut self, layer_no: usize, k_new: KVVector, v_new: KVVector) {
        let position_vector_k = &mut self.k[layer_no];
        position_vector_k.push(k_new);

        let position_vector_v = &mut self.v[layer_no];
        position_vector_v.push(v_new);
    }
}