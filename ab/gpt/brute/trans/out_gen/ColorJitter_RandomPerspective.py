import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.02, saturation=1.17, hue=0.02),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.84),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
