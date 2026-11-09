import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(49, 194, 240), padding_mode='constant'), transforms.RandomGrayscale(p=0.5), transforms.RandomPerspective(distortion_scale=0.17, p=0.3), transforms.RandomRotation(degrees=(-15, 15), p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])