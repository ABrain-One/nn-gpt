import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(100, 247, 151), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.89), ratio=(1.06, 2.56)),
    transforms.RandomGrayscale(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
