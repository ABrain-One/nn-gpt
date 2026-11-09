import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.77),
    transforms.RandomRotation(degrees=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
