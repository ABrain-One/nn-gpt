import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomAutocontrast(p=0.54),
    transforms.Pad(padding=4, fill=(92, 236, 245), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
