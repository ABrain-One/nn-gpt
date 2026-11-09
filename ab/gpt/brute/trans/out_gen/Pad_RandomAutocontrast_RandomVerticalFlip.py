import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(8, 76, 5), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.65),
    transforms.RandomVerticalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
