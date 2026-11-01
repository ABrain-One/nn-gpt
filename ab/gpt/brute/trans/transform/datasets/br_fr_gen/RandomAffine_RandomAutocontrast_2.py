import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=4, translate=(0.04, 0.11), scale=(1.11, 1.28), shear=(1.37, 8.53)),
    transforms.RandomAutocontrast(p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
