import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.21),
    transforms.RandomInvert(p=0.69),
    transforms.Pad(padding=5, fill=(185, 223, 163), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
