import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(196, 163, 161), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.62),
    transforms.RandomCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])