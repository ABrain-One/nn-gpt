import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.83), ratio=(1.01, 2.82)),
    transforms.RandomPosterize(bits=6, p=0.13),
    transforms.Pad(padding=5, fill=(224, 248, 148), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
