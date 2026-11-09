import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(250, 97, 204), padding_mode='expand'),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAutocontrast(p=0.36),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])