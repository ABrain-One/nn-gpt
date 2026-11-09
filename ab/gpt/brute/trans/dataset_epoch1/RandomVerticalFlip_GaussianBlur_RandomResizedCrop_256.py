import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.18),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.58, 1.96)),
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.97), ratio=(1.28, 2.72)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
