import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.98), ratio=(1.09, 2.66)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.45),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
