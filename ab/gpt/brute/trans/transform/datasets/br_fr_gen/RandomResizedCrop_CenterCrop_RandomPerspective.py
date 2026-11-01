import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.8), ratio=(1.27, 1.94)),
    transforms.CenterCrop(size=29),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
