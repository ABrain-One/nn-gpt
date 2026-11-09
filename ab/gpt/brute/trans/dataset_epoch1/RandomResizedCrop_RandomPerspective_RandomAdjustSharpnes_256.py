import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.95), ratio=(1.31, 1.8)),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.47),
    transforms.RandomAdjustSharpness(sharpness_factor=1.7, p=0.72),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
