import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=3, fill=(50, 91, 86), padding_mode='constant'),
        transforms.RandomVerticalFlip(p=0.81),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])