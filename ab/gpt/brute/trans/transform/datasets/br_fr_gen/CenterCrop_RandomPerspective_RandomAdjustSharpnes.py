import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.44),
    transforms.RandomAdjustSharpness(sharpness_factor=1.13, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
