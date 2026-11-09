import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(254, 155, 136), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.92), ratio=(0.98, 1.34)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.75, 1.61)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
