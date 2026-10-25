import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.86), ratio=(1.22, 1.62)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.81, p=0.74),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.33, 1.04)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
