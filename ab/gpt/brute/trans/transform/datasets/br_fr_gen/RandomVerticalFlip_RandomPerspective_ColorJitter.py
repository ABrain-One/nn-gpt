import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.52),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.34),
    transforms.ColorJitter(brightness=0.81, contrast=1.02, saturation=1.0, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
