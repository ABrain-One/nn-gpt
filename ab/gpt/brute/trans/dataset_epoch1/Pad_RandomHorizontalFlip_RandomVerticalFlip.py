import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(221, 57, 3), padding_mode='reflect'),
    transforms.RandomHorizontalFlip(p=0.82),
    transforms.RandomVerticalFlip(p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
