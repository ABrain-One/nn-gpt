import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.85), ratio=(1.16, 2.28)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.47, 1.33)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
