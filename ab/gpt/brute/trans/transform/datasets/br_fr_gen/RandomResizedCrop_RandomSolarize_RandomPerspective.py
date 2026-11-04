import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.84), ratio=(0.78, 1.92)),
    transforms.RandomSolarize(threshold=73, p=0.11),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
