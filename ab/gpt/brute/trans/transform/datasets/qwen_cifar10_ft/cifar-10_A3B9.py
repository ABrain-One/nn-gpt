import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(195, 213, 33), padding_mode='edge'), transforms.Grayscale(num_output_channels=3), transforms.RandomPerspective(distortion_scale=0.18, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])