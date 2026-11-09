import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.97), ratio=(1.12, 2.4)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.43, 1.24)),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
