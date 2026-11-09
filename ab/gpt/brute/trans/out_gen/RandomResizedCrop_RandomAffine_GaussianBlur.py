import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.85), ratio=(1.15, 1.83)),
    transforms.RandomAffine(degrees=11, translate=(0.05, 0.06), scale=(1.16, 1.69), shear=(1.7, 6.62)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.75, 1.19)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
