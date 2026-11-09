import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(67, 16, 189), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.78),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
