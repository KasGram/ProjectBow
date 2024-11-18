import torch

if torch.cuda.is_available():
    print("CUDA er tilgængelig!")
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA er ikke tilgængelig.")
