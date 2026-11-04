import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(118, 47, 196), padding_mode='symmetric'),
    transforms.RandomRotation(degrees=8),
    transforms.RandomVerticalFlip(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
