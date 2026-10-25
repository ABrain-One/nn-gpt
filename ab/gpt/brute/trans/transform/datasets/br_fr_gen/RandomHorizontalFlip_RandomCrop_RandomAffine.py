import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.37),
    transforms.RandomCrop(size=26),
    transforms.RandomAffine(degrees=20, translate=(0.17, 0.02), scale=(1.12, 1.26), shear=(1.4, 9.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
