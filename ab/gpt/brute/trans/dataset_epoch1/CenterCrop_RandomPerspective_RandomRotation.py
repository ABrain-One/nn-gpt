import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.22),
    transforms.RandomRotation(degrees=0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
