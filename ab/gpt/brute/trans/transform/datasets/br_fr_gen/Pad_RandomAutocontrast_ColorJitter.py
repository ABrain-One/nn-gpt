import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(36, 210, 118), padding_mode='symmetric'),
    transforms.RandomAutocontrast(p=0.61),
    transforms.ColorJitter(brightness=1.19, contrast=0.92, saturation=0.87, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
