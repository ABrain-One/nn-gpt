import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.77),
    transforms.RandomAffine(degrees=24, translate=(0.11, 0.18), scale=(1.12, 1.96), shear=(0.71, 9.11)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
