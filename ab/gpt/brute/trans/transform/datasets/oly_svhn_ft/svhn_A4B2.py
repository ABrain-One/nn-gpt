import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.71, 0.88), ratio=(1.23, 2.75)), transforms.RandomPerspective(distortion_scale=0.15, p=0.79), transforms.Pad(padding=0, fill=(148, 145, 46), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])