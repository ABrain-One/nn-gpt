import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.23),
    transforms.Pad(padding=4, fill=(49, 221, 101), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.84, contrast=1.14, saturation=1.13, hue=0.05),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
