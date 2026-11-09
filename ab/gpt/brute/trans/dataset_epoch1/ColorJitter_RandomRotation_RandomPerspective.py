import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=0.97, saturation=0.96, hue=0.02),
    transforms.RandomRotation(degrees=21),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
