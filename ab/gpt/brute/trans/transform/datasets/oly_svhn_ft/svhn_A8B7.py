import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(158, 173, 1)), transforms.Grayscale(num_output_channels=3), transforms.GaussianBlur(kernel_size=3, sigma=(0.89, 1.94)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])