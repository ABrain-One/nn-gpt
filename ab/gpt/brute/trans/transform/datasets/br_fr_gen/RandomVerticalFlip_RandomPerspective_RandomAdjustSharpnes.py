import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.86),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.12),
    transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
