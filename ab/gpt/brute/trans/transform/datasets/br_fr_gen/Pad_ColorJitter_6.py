import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(221, 17, 56), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.96, contrast=0.98, saturation=1.13, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
