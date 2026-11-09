<code>
import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([
        transforms.GaussianBlur(5, sigma=(0.5, 0.5, 0.5)),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Pad(1, fill=(212, 141, 188), padding_mode='reflect'),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</code>