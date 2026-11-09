import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.8), ratio=(1.32, 2.69)),
    transforms.RandomGrayscale(p=0.55),
    transforms.ColorJitter(brightness=0.86, contrast=0.85, saturation=0.83, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
