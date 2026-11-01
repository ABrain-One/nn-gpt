import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(172, 195, 133), padding_mode='edge'), transforms.RandomSolarize(threshold=100, p=0.24), transforms.RandomRotation(degrees=10), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])