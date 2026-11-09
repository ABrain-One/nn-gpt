import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(4, 147, 238), padding_mode='symmetric'),
    transforms.RandomInvert(p=0.62),
    transforms.ColorJitter(brightness=0.95, contrast=1.19, saturation=0.83, hue=0.04),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
