import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=11, translate=(0.13, 0.0), scale=(1.11, 1.75), shear=(0.62, 5.14)),
    transforms.CenterCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
