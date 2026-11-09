import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.88), ratio=(1.16, 2.36)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.99, 1.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
