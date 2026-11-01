import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.88, 1.01)),
    transforms.RandomAffine(degrees=5, translate=(0.08, 0.05), scale=(1.18, 1.25), shear=(0.13, 6.13)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
