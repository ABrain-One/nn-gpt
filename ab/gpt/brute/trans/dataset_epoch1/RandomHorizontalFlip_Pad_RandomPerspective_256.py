import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.71),
    transforms.Pad(padding=2, fill=(3, 103, 167), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.88),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
