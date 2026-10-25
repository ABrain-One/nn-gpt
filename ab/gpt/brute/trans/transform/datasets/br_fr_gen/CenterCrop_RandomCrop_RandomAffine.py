import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomCrop(size=25),
    transforms.RandomAffine(degrees=22, translate=(0.11, 0.17), scale=(1.11, 1.99), shear=(2.98, 7.53)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
