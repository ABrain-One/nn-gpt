import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.91, contrast=1.06, saturation=0.86, hue=0.02),
    transforms.RandomGrayscale(p=0.35),
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
