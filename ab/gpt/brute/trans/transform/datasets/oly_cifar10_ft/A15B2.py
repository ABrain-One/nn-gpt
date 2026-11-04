import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=0, fill=(217, 4, 28), padding_mode='edge'),
        transforms.RandomPerspective(distortion_scale=0.17, p=0.5),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])