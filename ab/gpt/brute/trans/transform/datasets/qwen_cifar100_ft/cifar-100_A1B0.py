import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(133, 168, 134), padding_mode='symmetric'), transforms.RandomAutocontrast(p=0.82), transforms.GaussianBlur(kernel_size=5, sigma=(0.28, 1.06)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])