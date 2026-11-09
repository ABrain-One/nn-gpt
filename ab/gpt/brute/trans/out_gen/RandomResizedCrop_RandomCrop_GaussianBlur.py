import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.32, 2.84)),
    transforms.RandomCrop(size=31),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.54, 1.66)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
