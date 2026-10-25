import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(62, 228, 38), padding_mode='constant'),
    transforms.CenterCrop(size=30),
    transforms.ColorJitter(brightness=0.89, contrast=1.16, saturation=1.07, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
