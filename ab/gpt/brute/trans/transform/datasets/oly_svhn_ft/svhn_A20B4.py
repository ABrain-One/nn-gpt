import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(198, 105, 45), padding_mode='constant'), transforms.RandomPerspective(distortion_scale=0.21, p=0.28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])