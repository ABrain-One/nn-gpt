import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomAffine(degrees=21, translate=(0.08, 0.17), scale=(1.04, 1.39), shear=(0.9, 9.68)),
    transforms.RandomHorizontalFlip(p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
