import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(10, 10, device=device, dtype=torch.bfloat16)
    print(f"bfloat16 tensor on GPU: {x}")
else:
    print("CUDA is not available")