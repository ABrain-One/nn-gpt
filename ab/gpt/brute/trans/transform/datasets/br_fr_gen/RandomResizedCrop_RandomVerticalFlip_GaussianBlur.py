import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.95), ratio=(0.76, 2.08)),
    transforms.RandomVerticalFlip(p=0.38),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.35, 1.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
