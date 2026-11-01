import torch
import torchvision.transforms as transforms

def transform(norm):
    mean, std = norm
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.6, 1.3333), ratio=(0.8, 1.2)), transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), transforms.RandomGrayscale(p=0.3), transforms.RandomPerspective(distortion_scale=0.3, p=0.3), transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])