import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(173, 16, 189), padding_mode='edge'), transforms.RandomPerspective(distortion_scale=0.18, p=0.46), transforms.RandomEqualize(p=0.52), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])