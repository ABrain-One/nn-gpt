import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.74, 1.67)),
    transforms.RandomAffine(degrees=13, translate=(0.1, 0.17), scale=(1.08, 1.36), shear=(0.38, 7.41)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
