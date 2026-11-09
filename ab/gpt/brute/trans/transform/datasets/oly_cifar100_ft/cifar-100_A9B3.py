import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(175, 151, 253), padding_mode='symmetric'), transforms.RandomSolarize(threshold=190, p=0.22), transforms.RandomPerspective(distortion_scale=0.11, p=0.6), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])