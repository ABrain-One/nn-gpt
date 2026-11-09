import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.98, 1.42)),
    transforms.RandomAffine(degrees=25, translate=(0.04, 0.14), scale=(0.86, 1.51), shear=(4.97, 9.7)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
