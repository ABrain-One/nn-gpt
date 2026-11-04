import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.85), transforms.RandomPerspective(distortion_scale=0.19, p=0.55), transforms.Pad(padding=0, fill=(121, 135, 241), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])