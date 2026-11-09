import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomEqualize(p=0.2),
    transforms.RandomAffine(degrees=28, translate=(0.13, 0.11), scale=(0.84, 1.85), shear=(4.15, 7.92)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
