import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(31, 112, 50), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.RandomCrop(size=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
