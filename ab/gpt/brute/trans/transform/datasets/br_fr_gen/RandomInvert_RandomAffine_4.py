import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.73),
    transforms.RandomAffine(degrees=23, translate=(0.17, 0.02), scale=(0.8, 1.85), shear=(1.27, 6.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
