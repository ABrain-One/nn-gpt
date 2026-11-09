import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.97), ratio=(0.96, 2.65)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.5),
    transforms.RandomInvert(p=0.9),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
