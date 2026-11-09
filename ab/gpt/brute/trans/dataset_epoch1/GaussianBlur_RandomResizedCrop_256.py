import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.98, 1.03)),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.9), ratio=(1.02, 1.6)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
