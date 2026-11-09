import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.75, 0.94), ratio=(1.05, 1.53)), transforms.RandomGrayscale(p=0.43), transforms.RandomPerspective(distortion_scale=0.18, p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])