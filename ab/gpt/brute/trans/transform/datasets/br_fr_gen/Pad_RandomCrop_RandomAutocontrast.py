import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(43, 79, 30), padding_mode='symmetric'),
    transforms.RandomCrop(size=25),
    transforms.RandomAutocontrast(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
