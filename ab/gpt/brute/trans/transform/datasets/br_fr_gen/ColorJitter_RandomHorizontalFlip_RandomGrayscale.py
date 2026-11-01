import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=0.88, saturation=1.02, hue=0.1),
    transforms.RandomHorizontalFlip(p=0.48),
    transforms.RandomGrayscale(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
