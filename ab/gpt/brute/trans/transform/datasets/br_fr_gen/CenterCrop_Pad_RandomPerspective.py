import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.Pad(padding=5, fill=(121, 91, 37), padding_mode='reflect'),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
