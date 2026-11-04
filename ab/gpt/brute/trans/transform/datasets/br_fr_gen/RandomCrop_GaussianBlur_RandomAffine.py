import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.16, 1.07)),
    transforms.RandomAffine(degrees=20, translate=(0.08, 0.19), scale=(0.85, 1.29), shear=(2.83, 9.64)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
