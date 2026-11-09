import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.55),
    transforms.Pad(padding=0, fill=(178, 57, 144), padding_mode='reflect'),
    transforms.RandomCrop(size=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
