import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.21),
    transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=1.15, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
