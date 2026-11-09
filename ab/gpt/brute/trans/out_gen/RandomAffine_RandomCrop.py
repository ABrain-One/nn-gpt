import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.07, 0.11), scale=(1.16, 1.76), shear=(4.98, 9.24)),
    transforms.RandomCrop(size=27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
