import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(35, 56, 251), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.09, contrast=1.17, saturation=1.13, hue=0.02),
    transforms.RandomGrayscale(p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
