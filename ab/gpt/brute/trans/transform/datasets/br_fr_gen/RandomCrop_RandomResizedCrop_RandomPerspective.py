import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.94), ratio=(1.29, 2.21)),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
