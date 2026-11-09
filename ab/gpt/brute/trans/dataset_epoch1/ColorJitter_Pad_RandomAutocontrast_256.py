import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=0.95, saturation=0.92, hue=0.07),
    transforms.Pad(padding=5, fill=(227, 82, 213), padding_mode='symmetric'),
    transforms.RandomAutocontrast(p=0.42),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
