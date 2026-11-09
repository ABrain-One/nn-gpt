import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=1.19, saturation=0.95, hue=0.0),
    transforms.CenterCrop(size=25),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.42),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
