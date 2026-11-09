import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(164, 148, 50), padding_mode='constant'),
    transforms.ColorJitter(brightness=1.06, contrast=1.12, saturation=0.96, hue=0.08),
    transforms.RandomHorizontalFlip(p=0.27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
