import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(35, 20, 218), padding_mode='reflect'),
    transforms.ColorJitter(brightness=1.17, contrast=0.89, saturation=0.96, hue=0.01),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
