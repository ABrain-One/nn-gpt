import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomEqualize(p=0.12), transforms.GaussianBlur(kernel_size=5, sigma=(0.91, 1.33)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])