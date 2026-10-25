import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.ColorJitter(brightness=0.86, contrast=0.93, saturation=1.16, hue=0.04),
    transforms.RandomGrayscale(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
