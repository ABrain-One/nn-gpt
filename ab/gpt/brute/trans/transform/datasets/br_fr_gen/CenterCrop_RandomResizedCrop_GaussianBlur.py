import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.99), ratio=(1.13, 2.8)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.54, 1.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
