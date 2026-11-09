import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(118, 110, 64), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.74),
    transforms.RandomInvert(p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
