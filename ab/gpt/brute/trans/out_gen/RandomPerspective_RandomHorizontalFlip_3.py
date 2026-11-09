import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.23, p=0.54),
    transforms.RandomHorizontalFlip(p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
