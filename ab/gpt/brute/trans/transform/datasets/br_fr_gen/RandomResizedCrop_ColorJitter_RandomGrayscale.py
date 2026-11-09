import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.99), ratio=(0.82, 1.69)),
    transforms.ColorJitter(brightness=1.16, contrast=1.04, saturation=1.15, hue=0.03),
    transforms.RandomGrayscale(p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
