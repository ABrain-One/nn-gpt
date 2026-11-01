import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.ColorJitter(brightness=0.81, contrast=1.04, saturation=0.85, hue=0.03),
    transforms.RandomGrayscale(p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
