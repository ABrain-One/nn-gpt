import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.19),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.86), ratio=(1.1, 1.99)),
    transforms.RandomPerspective(distortion_scale=0.24, p=0.4),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
