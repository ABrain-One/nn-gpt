import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.0, contrast=0.97, saturation=0.86, hue=0.05),
    transforms.RandomCrop(size=27),
    transforms.RandomGrayscale(p=0.27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
