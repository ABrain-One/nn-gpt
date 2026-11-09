import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.34),
    transforms.RandomAffine(degrees=23, translate=(0.13, 0.18), scale=(1.04, 1.98), shear=(4.51, 5.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
