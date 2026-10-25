import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.78),
    transforms.RandomAffine(degrees=2, translate=(0.13, 0.14), scale=(1.13, 1.83), shear=(0.34, 5.26)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
