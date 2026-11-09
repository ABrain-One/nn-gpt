import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.92), ratio=(1.22, 2.59)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.25, 1.33)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
