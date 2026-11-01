import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.16),
    transforms.RandomAffine(degrees=16, translate=(0.15, 0.15), scale=(1.1, 1.92), shear=(2.52, 7.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
