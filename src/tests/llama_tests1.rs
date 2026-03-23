pub fn llama_tests1() {

    let input_norm_w = &llama_model.layers[0].input_norm_w;
    let q_proj_gpu = &llama_model.layers[0].q_proj_w;
    let k_proj_gpu = &llama_model.layers[0].k_proj_w;
    let v_proj_gpu = &llama_model.layers[0].v_proj_w;
    let gate_proj_gpu = &llama_model.layers[0].gate_proj_w;


    let token_id = 1usize;
    let token_start = token_id * llama_model.config.hidden_size;
    let embed_token = &llama_model.embed_weights[token_start..token_start + llama_model.config.hidden_size];
    println!("embed_token first 5: {:?}", &embed_token[0..5]);

    let normed = rms_norm(embed_token, &input_norm_w, 1e-5);
    println!("rms_norm {:?}", &normed[0..5]);

    let q = matmul_transposed(&normed, &q_proj_gpu, llama_model.config.hidden_size, &gpu);
    println!("q {:?}", &q[0..5]);

    let gate = matmul_transposed(&normed, &gate_proj_gpu, llama_model.config.intermediate_size, &gpu);
    let silu_gate = silu(&gate);
    println!("silu_gate {:?}", &silu_gate[0..5]);

    let (cos, sin) = compute_rope_freqs(llama_model.config.head_dim, 1, llama_model.config.rope_theta, llama_model.config.head_dim/2);
    println!("cos {:?}", &cos[0..5]);
    println!("sin {:?}", &sin[0..5]);

    let q_rope = apply_rope(&q, &cos, &sin, llama_model.config.head_dim);
    println!("q_rope {:?}", &q_rope[0..5]);

    let k = matmul_transposed(&normed, &k_proj_gpu, llama_model.config.kv_dim, &gpu);
    let k_rope = apply_rope(&k, &cos, &sin, llama_model.config.head_dim );
    println!("k_rope {:?}", &k_rope[0..5]);

    let v = matmul_transposed(&normed, v_proj_gpu, llama_model.config.kv_dim, &gpu);
    println!("v[:5]: {:?}", &v[0..5]);

    let repeats = llama_model.config.heads_to_kvheads;
    let mut v_expanded = vec![0.0f32; llama_model.config.hidden_size];
    for kv_h in 0..llama_model.config.num_key_value_heads {
        let src = kv_h * llama_model.config.head_dim;
        let dst_start = src * repeats;
        for r in 0..repeats {
            let dst = dst_start + r * llama_model.config.head_dim;
            let v_slice = &v[src..src + llama_model.config.head_dim];
            v_expanded[dst..dst+llama_model.config.head_dim].copy_from_slice(v_slice);
        }
    }

    let o_proj_gpu = &llama_model.layers[0].o_proj_w;
    let attn_out = matmul_transposed(&v_expanded, o_proj_gpu, llama_model.config.hidden_size, &gpu);
    println!("attn_out {:?}", &attn_out[0..5]);

    let mut hidden: Vecf32 = embed_token.to_vec();
    add_to_vector(&mut hidden, &attn_out);
    let post_norm_w = &llama_model.layers[0].post_norm_w;
    let normed2 = rms_norm(&hidden, post_norm_w, llama_model.config.rms_norm_eps);
    let gate = matmul_transposed(&normed2, &llama_model.layers[0].gate_proj_w, llama_model.config.intermediate_size, &gpu);
    let up = matmul_transposed(&normed2, &llama_model.layers[0].up_proj_w, llama_model.config.intermediate_size, &gpu);
    let silu_gate = silu(&gate);

    let mut ffn_act = vec![0.0f32; llama_model.config.intermediate_size];

    for i in 0..llama_model.config.intermediate_size {
        ffn_act[i] = silu_gate[i] * up[i];
    }

    let ffn_out = matmul_transposed(&ffn_act, &llama_model.layers[0].down_proj_w, llama_model.config.hidden_size, &gpu);
    add_to_vector(&mut hidden, &ffn_out);

    println!("layer0_out[:5]: {:?}", &hidden[0..5]);
}