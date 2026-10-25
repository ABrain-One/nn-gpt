import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.46),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 1.0), ratio=(1.04, 2.21)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.16, 1.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
