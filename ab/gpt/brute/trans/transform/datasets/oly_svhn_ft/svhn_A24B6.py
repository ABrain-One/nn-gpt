import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPerspective(distortion_scale=0.13, p=0.66), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.2931, 0.2821, 0.2904], std=[0.2801, 0.2821, 0.2801])])