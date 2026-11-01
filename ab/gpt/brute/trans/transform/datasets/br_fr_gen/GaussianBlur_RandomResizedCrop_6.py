import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.54, 1.15)),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.88), ratio=(0.77, 2.33)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
