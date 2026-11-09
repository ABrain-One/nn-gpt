import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.42),
    transforms.RandomAffine(degrees=2, translate=(0.11, 0.04), scale=(1.14, 1.93), shear=(0.92, 6.22)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
