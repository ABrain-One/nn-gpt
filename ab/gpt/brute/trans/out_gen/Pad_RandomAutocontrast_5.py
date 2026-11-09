import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(255, 88, 41), padding_mode='symmetric'),
    transforms.RandomAutocontrast(p=0.36),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
