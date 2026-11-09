import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.62),
    transforms.RandomGrayscale(p=0.8),
    transforms.RandomAffine(degrees=28, translate=(0.2, 0.05), scale=(1.07, 1.29), shear=(1.63, 7.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
