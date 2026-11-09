import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.18), transforms.GaussianBlur(kernel_size=5, sigma=(0.92, 1.29)), transforms.CenterCrop(size=27), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])