import torch

def check_gpu():
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
