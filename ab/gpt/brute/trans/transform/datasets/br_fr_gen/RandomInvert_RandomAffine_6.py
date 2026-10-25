import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.55),
    transforms.RandomAffine(degrees=9, translate=(0.19, 0.01), scale=(1.15, 1.65), shear=(0.99, 9.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
