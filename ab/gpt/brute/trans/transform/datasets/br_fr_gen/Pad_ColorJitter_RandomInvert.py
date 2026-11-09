import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(79, 63, 185), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.99, contrast=1.13, saturation=1.13, hue=0.04),
    transforms.RandomInvert(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
