import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(" ".join(f"{v:.4f}" for v in torch.softmax(x, dim=0).tolist()))
