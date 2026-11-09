import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.Pad(padding=1, fill=(232, 14, 41), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.83),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
