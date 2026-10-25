import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.94), ratio=(0.94, 2.88)),
    transforms.RandomEqualize(p=0.48),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.57, 1.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
