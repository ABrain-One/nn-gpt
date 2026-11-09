import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.87), ratio=(1.01, 2.28)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.89, 1.32)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
