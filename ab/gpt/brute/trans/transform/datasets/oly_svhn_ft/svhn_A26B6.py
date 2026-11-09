import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(12, 216, 75), padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.26, p=0.46), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])