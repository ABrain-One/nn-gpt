import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.19, 0.13), scale=(1.04, 1.78), shear=(0.13, 5.24)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.77, 1.04)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
