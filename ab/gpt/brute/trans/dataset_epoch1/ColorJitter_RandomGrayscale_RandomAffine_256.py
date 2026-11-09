import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.9, contrast=1.12, saturation=0.85, hue=0.06),
    transforms.RandomGrayscale(p=0.42),
    transforms.RandomAffine(degrees=2, translate=(0.03, 0.17), scale=(1.09, 1.85), shear=(3.62, 9.47)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
