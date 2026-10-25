import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(33, 134, 101), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.73),
    transforms.RandomInvert(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])