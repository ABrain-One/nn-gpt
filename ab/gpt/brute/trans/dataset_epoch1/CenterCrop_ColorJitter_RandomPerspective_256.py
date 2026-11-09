import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.ColorJitter(brightness=1.12, contrast=0.88, saturation=0.81, hue=0.04),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.88),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
