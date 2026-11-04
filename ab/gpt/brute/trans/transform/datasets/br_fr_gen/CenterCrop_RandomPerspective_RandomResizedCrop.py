import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.27),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.86), ratio=(0.81, 1.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
