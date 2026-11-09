import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.91, contrast=0.95, saturation=1.09, hue=0.01),
    transforms.Pad(padding=2, fill=(219, 240, 27), padding_mode='edge'),
    transforms.RandomEqualize(p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
