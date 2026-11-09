import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.62),
    transforms.Pad(padding=2, fill=(12, 248, 143), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
