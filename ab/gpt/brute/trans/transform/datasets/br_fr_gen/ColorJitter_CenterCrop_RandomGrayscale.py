import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.94, contrast=0.95, saturation=1.18, hue=0.06),
    transforms.CenterCrop(size=25),
    transforms.RandomGrayscale(p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
