import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(25), transforms.RandomPerspective(distortion_scale=0.2, p=0.5), transforms.ColorJitter(0.5, 0.5, 0.5, 0.05), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])