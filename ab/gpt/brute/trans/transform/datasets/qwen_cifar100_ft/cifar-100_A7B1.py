import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomPerspective(distortion_scale=0.11, p=0.23), transforms.RandomPosterize(bits=5, p=0.46), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])