import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomAdjustSharpness(sharpness_factor=0.92, p=0.32),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.29, 1.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
