import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.Pad(padding=0, fill=(111, 102, 49), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.76),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
