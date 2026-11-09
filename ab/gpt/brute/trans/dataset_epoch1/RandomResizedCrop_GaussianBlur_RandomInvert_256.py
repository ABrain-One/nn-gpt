import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.97), ratio=(0.8, 2.25)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.53, 1.31)),
    transforms.RandomInvert(p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
