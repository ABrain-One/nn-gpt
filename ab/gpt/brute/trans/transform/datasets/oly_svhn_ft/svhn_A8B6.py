import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPerspective(distortion_scale=0.12, p=0.72), transforms.RandomGrayscale(p=0.22), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])