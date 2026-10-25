import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomGrayscale(p=0.24),
    transforms.RandomAffine(degrees=8, translate=(0.1, 0.04), scale=(1.01, 1.78), shear=(4.61, 7.5)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
