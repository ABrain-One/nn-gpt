import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(152, 101, 138), padding_mode='constant'), transforms.RandomPerspective(distortion_scale=0.17, p=0.57), transforms.RandomHorizontalFlip(p=0.35), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])