import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(215, 9, 115), padding_mode='constant'), transforms.RandomRotation(degrees=13), transforms.RandomPerspective(distortion_scale=0.25, p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])