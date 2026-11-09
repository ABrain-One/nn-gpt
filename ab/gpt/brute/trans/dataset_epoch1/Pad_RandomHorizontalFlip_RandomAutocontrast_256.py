import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(179, 183, 37), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.47),
    transforms.RandomAutocontrast(p=0.27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
