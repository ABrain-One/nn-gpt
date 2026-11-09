import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(135, 138, 251), padding_mode='reflect'),
    transforms.RandomRotation(degrees=7),
    transforms.RandomGrayscale(p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
