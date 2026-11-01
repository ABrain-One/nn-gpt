import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=3, sigma=(0.88, 1.28)),
    transforms.RandomAffine(degrees=11, translate=(0.15, 0.15), scale=(0.94, 1.32), shear=(2.21, 7.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
