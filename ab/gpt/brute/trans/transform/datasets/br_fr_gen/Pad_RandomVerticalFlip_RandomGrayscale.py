import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(29, 29, 78), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.8),
    transforms.RandomGrayscale(p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
