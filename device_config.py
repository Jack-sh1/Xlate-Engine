import torch

def get_device():
    """
    Determine the best available device for PyTorch.
    Order: MPS (Mac) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
print(f"Detected hardware acceleration device: {DEVICE}")
