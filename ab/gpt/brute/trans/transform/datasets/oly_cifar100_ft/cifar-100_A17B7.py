import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(173, 250, 228), padding_mode='symmetric'), transforms.RandomPerspective(distortion_scale=0.14, p=0.16), transforms.RandomPosterize(bits=7, p=0.32), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])