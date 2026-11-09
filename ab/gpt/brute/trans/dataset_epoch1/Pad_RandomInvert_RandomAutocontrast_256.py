import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(99, 171, 214), padding_mode='edge'),
    transforms.RandomInvert(p=0.82),
    transforms.RandomAutocontrast(p=0.13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
