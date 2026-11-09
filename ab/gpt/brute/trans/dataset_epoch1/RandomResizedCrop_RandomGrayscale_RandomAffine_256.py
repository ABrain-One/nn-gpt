import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.82), ratio=(1.15, 2.02)),
    transforms.RandomGrayscale(p=0.76),
    transforms.RandomAffine(degrees=22, translate=(0.13, 0.0), scale=(1.16, 1.85), shear=(1.43, 8.61)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
