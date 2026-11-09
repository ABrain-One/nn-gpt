import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(54, 117, 38), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.88),
    transforms.RandomEqualize(p=0.53),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])