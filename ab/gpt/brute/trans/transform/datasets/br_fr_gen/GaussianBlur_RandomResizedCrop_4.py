import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.14, 1.55)),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.94), ratio=(0.76, 2.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
