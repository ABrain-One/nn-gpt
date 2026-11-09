import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.88),
    transforms.RandomAdjustSharpness(sharpness_factor=1.16, p=0.68),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
