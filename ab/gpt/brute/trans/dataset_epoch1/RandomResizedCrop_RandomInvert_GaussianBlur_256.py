import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.9), ratio=(0.9, 2.17)),
    transforms.RandomInvert(p=0.34),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.45)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
