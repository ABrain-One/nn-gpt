import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.13),
    transforms.RandomEqualize(p=0.58),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
