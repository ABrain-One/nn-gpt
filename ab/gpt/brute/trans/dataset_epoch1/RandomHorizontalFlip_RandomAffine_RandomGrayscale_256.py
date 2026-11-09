import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.48),
    transforms.RandomAffine(degrees=21, translate=(0.09, 0.13), scale=(1.17, 1.55), shear=(4.95, 6.4)),
    transforms.RandomGrayscale(p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
