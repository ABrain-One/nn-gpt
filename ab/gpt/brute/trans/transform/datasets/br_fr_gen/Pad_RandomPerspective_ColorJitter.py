import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(161, 59, 204), padding_mode='reflect'),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.18),
    transforms.ColorJitter(brightness=0.99, contrast=0.81, saturation=1.01, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
