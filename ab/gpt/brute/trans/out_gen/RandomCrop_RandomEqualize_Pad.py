import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomEqualize(p=0.75),
    transforms.Pad(padding=1, fill=(149, 38, 225), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
