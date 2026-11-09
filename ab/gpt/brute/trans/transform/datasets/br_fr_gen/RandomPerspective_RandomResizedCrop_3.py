import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.27, p=0.54),
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(0.77, 1.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
