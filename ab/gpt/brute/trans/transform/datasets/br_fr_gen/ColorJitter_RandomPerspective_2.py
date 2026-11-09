import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.81, contrast=0.93, saturation=1.16, hue=0.01),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
