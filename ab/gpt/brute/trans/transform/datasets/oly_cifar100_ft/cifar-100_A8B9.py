import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.66), transforms.RandomPosterize(bits=8, p=0.78), transforms.RandomRotation(degrees=14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.481, 0.461, 0.721], std=[0.298, 0.292, 0.771])])