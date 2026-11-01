import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(64), transforms.CenterCrop(48), transforms.RandomVerticalFlip(p=0.14), transforms.Pad(padding=5, fill=(27, 95, 165), padding_mode='constant'), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])