import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=0.91, saturation=0.94, hue=0.01),
    transforms.RandomGrayscale(p=0.18),
    transforms.Pad(padding=0, fill=(187, 218, 224), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
