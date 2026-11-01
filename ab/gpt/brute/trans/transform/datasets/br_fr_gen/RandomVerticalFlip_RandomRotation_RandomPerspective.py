import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.21),
    transforms.RandomRotation(degrees=11),
    transforms.RandomPerspective(distortion_scale=0.16, p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
