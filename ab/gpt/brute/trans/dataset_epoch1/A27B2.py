import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(221, 57, 39), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])