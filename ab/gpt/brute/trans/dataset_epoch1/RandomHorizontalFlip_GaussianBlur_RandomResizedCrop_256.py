import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.24),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.77, 1.91)),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.82), ratio=(1.32, 2.05)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
