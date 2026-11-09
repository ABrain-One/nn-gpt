import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.47),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.99), ratio=(1.12, 1.61)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
