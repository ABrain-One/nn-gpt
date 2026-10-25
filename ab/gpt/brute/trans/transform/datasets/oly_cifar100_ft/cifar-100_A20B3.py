import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(160, 240, 152), padding_mode='edge'), transforms.RandomSolarize(threshold=221, p=0.15), transforms.RandomPerspective(distortion_scale=0.14, p=0.67), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])