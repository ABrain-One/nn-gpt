import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.Pad(padding=0, fill=(34, 178, 96), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
