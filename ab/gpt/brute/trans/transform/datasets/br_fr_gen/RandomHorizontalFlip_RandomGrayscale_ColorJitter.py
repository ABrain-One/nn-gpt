import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.27),
    transforms.RandomGrayscale(p=0.3),
    transforms.ColorJitter(brightness=1.12, contrast=0.84, saturation=0.83, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
