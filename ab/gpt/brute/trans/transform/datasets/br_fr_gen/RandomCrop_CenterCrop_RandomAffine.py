import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.CenterCrop(size=28),
    transforms.RandomAffine(degrees=5, translate=(0.04, 0.06), scale=(0.93, 1.39), shear=(1.48, 5.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
