import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(173, 219, 42), padding_mode='edge'),
    transforms.RandomAutocontrast(p=0.74),
    transforms.RandomHorizontalFlip(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
