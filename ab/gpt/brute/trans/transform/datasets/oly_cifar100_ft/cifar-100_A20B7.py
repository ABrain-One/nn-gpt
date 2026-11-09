import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.15), transforms.Pad(padding=0, fill=(69, 161, 204), padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.27, p=0.19), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.758, 0.82, 0.157), (0.0395, 0.0337, 0.286))])