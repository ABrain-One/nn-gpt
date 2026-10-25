import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.Grayscale(num_output_channels=3), transforms.GaussianBlur(kernel_size=3, sigma=(0.13, 1.19)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])