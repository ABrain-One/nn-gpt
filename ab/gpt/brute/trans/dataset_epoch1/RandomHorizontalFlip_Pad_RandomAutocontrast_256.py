import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.59),
    transforms.Pad(padding=2, fill=(255, 79, 14), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.55),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
