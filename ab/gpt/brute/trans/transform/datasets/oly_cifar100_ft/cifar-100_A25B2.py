import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(247, 131, 125), padding_mode='constant'), transforms.RandomPerspective(distortion_scale=0.18, p=0.77), transforms.RandomPosterize(bits=7, p=0.54), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])