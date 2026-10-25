import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(26, 14, 114), padding_mode='constant'),
    transforms.RandomEqualize(p=0.52),
    transforms.ColorJitter(brightness=0.9, contrast=0.91, saturation=1.16, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
