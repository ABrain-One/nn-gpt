import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(10, 194, 202), padding_mode='reflect'),
    transforms.ColorJitter(brightness=0.8, contrast=1.1, saturation=1.1, hue=0.07),
    transforms.CenterCrop(size=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
