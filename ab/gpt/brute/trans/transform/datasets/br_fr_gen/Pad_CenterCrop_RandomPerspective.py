import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(12, 9, 63), padding_mode='reflect'),
    transforms.CenterCrop(size=31),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
