import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.79),
    transforms.RandomAdjustSharpness(sharpness_factor=1.34, p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
