import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.57),
    transforms.ColorJitter(brightness=0.82, contrast=1.13, saturation=1.11, hue=0.06),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.22),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
