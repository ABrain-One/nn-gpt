import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(217, 108, 223), padding_mode='constant'), transforms.RandomAutocontrast(p=0.52), transforms.GaussianBlur(kernel_size=3, sigma=(0.53, 1.74)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])