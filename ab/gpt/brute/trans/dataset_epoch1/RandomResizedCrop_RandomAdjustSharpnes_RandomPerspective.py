import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.9), ratio=(0.76, 2.53)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.1, p=0.37),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
