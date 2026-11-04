import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.88, 1.81)),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.93), ratio=(1.32, 1.45)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
