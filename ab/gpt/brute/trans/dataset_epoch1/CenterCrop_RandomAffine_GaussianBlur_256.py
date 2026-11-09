import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomAffine(degrees=7, translate=(0.07, 0.02), scale=(1.15, 1.64), shear=(3.02, 6.87)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.85, 1.45)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
