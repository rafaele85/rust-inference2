pub fn combine_embedding_weights(wte_row: &[f32], wpe_row: &[f32]) -> Vec<f32> {
    let result = wte_row.iter()
        .zip(wpe_row.iter())
        .map(|(a, b)| a + b)
        .collect::<Vec<f32>>();

    result
}
