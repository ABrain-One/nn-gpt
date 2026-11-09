import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomAffine(degrees=11, translate=(0.01, 0.13), scale=(1.05, 1.76), shear=(3.08, 6.1)),
    transforms.RandomEqualize(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
