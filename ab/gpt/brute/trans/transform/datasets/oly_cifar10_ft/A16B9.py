import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=1, fill=(255, 58, 95), padding_mode='constant'),
        transforms.RandomAutoquantize(p=0.68),
        transforms.RandomAffine(degrees=13, translate=(0.03, 0.14), scale=(1.15, 1.95), shear=(3.02, 6.81)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])