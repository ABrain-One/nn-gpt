import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.55),
    transforms.RandomGrayscale(p=0.12),
    transforms.RandomAffine(degrees=29, translate=(0.09, 0.05), scale=(0.88, 1.21), shear=(3.02, 6.12)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
