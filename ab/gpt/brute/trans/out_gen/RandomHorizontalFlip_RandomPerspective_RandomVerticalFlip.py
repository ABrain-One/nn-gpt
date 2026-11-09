import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.63),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.5),
    transforms.RandomVerticalFlip(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
