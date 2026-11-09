import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.02, saturation=0.95, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.01, 0.16), scale=(0.96, 1.5), shear=(1.28, 9.87)),
    transforms.RandomGrayscale(p=0.78),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
