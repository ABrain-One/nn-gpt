import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.44),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.46),
    transforms.ColorJitter(brightness=1.06, contrast=1.01, saturation=1.07, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
