import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(151, 66, 232), padding_mode='edge'),
    transforms.RandomAutocontrast(p=0.88),
    transforms.RandomEqualize(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
