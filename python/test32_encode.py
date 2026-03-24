from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(tok.encode("Hello world", add_special_tokens=False))


tokens = tok.tokenize("Hello world")
print("tokens:", tokens)
print("bytes:", [t.encode() for t in tokens])

vocab = tok.get_vocab()
for k in list(vocab.keys())[:10]:
    print(repr(k), vocab[k])


print("▁Hello" in vocab, vocab.get("▁Hello"))
print("▁world" in vocab, vocab.get("▁world"))   
