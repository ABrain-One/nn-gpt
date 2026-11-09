import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(167, 167, 241), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.71),
    transforms.RandomRotation(degrees=13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
