import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=2, fill=(2, 160, 136), padding_mode='reflect'),
        transforms.RandomAdjustSharpness(sharpness_factor=1.84, p=0.38),
        transforms.RandomCrop(size=27),
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.48, std=0.29, inplace=True)
])