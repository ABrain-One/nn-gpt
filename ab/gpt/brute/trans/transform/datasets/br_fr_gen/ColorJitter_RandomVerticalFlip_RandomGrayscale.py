import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.07, contrast=1.09, saturation=1.0, hue=0.05),
    transforms.RandomVerticalFlip(p=0.41),
    transforms.RandomGrayscale(p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
