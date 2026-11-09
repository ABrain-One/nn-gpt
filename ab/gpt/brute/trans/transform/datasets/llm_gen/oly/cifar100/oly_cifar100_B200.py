import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomPerspective(distortion_scale=0.25, p=0.5), transforms.Pad(padding=1, fill=(108, 134, 41), padding_mode='constant'), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])