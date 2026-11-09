import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=27, translate=(0.18, 0.01), scale=(1.16, 1.36), shear=(3.12, 7.9)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.35, 1.1)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
