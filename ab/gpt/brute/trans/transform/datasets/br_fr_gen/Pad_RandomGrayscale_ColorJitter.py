import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(195, 205, 243), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.44),
    transforms.ColorJitter(brightness=0.93, contrast=1.19, saturation=1.12, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
