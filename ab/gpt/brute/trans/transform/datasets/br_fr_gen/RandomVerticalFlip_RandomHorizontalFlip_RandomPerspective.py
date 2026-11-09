import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.14),
    transforms.RandomHorizontalFlip(p=0.64),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.65),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
