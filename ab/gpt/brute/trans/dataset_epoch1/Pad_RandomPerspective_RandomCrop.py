import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(73, 104, 18), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.62),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
