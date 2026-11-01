import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=3, fill=(195, 74, 222), padding_mode='constant'),
        transforms.RandomPosterize(bits=7, p=0.4),
        transforms.RandomResizedCrop(size=32, scale=(0.73, 0.97), ratio=(1.09, 1.44)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])