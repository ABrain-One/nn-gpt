import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPerspective(distortion_scale=0.28, p=0.89), transforms.ColorJitter(brightness=0.86, contrast=0.93, saturation=1.16, hue=0.04), transforms.RandomGrayscale(p=0.29), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])