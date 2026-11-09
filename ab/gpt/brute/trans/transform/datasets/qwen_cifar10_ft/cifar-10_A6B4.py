import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.RandomPerspective(distortion_scale=0.26, p=0.52), transforms.RandomGrayscale(p=0.38), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])