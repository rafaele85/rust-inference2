# TODO

## Hardcoded / Temporary

- [ ] Single token input (token ID 464, position 0) — extend to handle token sequences
- [ ] epsilon = 1e-5 hardcoded in layer_norm — read from config.json (layer_norm_epsilon)
- [X] Only block 0 weights loaded — loop through all 12 blocks
- [ ] No tokenizer — implement BPE encoding/decoding
- [ ] No generation loop — implement greedy autoregressive decoding
- [ ] Model path hardcoded as string — accept as CLI argument
- [ ] Flat f32 slices have no shape info — introduce Tensor struct to carry shape with data
- [X] Attention is single-token only — implement multi-head scaled dot-product attention for sequences
- [ ] Read dim, n_heads, vocab_size, ffn_dim from config.json instead of hardcoding