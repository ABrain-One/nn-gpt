import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.24),
    transforms.RandomVerticalFlip(p=0.68),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
