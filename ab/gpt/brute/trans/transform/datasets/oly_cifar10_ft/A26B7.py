import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=3, fill=(41, 101, 116), padding_mode='edge'),
        transforms.RandomAdjustSharpness(sharpness_factor=0.79, p=0.73),
        transforms.RandomHorizontalFlip(p=0.63),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])