import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.9), ratio=(0.97, 1.6)),
    transforms.RandomEqualize(p=0.15),
    transforms.RandomAffine(degrees=0, translate=(0.19, 0.11), scale=(1.11, 1.76), shear=(0.72, 5.83)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
