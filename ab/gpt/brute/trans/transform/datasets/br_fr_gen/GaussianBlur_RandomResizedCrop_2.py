import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.84)),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.89), ratio=(0.96, 2.04)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
