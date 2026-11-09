import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomAdjustSharpness(sharpness_factor=0.85, p=0.73),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
