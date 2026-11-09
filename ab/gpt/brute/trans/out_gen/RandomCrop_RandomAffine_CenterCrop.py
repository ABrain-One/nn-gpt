import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomAffine(degrees=22, translate=(0.08, 0.0), scale=(0.86, 1.64), shear=(4.74, 9.46)),
    transforms.CenterCrop(size=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
