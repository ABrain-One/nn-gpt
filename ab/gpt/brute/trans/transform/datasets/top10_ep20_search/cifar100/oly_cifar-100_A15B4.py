import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(173, 250, 228), padding_mode='symmetric'), transforms.RandomPerspective(distortion_scale=0.19, p=0.19), transforms.RandomPosterize(bits=7, p=0.66), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])