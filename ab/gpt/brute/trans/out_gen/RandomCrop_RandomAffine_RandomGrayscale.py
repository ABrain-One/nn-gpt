import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomAffine(degrees=14, translate=(0.08, 0.03), scale=(0.94, 1.96), shear=(4.16, 8.05)),
    transforms.RandomGrayscale(p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
