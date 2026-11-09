import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomHorizontalFlip(p=0.46),
    transforms.RandomAffine(degrees=19, translate=(0.1, 0.08), scale=(0.84, 1.21), shear=(2.23, 7.0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
