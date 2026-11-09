import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(148, 121, 147), padding_mode='symmetric'), transforms.RandomPerspective(distortion_scale=0.17, p=0.37), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])