import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.85),
        transforms.Pad(padding=2, fill=(157, 161, 41), padding_mode='edge'),
        transforms.RandomPerspective(distortion_scale=0.19, p=0.5),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])