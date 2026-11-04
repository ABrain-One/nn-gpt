import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=18, translate=(0.08, 0.15), scale=(0.85, 1.73), shear=(2.84, 5.13)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.95, 1.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
