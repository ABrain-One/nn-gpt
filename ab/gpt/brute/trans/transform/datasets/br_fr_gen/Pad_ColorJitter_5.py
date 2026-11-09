import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(227, 54, 87), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=1.18, contrast=1.04, saturation=1.11, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
