import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.86), ratio=(1.12, 1.37)),
    transforms.RandomAffine(degrees=27, translate=(0.01, 0.15), scale=(1.14, 1.5), shear=(0.85, 7.05)),
    transforms.RandomGrayscale(p=0.89),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
