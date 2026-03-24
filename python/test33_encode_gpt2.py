from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.encode("The capital of France is", add_special_tokens=False))
print(tok.tokenize("The capital of France is"))
print(repr(tok.tokenize("The capital of France is")))
print(tok.encode("hello", add_special_tokens=False))
