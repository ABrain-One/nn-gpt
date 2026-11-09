import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.83),
    transforms.RandomAffine(degrees=13, translate=(0.18, 0.02), scale=(0.93, 1.71), shear=(2.94, 8.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
