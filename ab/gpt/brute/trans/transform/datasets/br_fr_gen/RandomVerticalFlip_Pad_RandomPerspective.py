import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.48),
    transforms.Pad(padding=1, fill=(221, 97, 106), padding_mode='constant'),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.41),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
