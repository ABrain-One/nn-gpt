import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(165, 187, 243), padding_mode='constant'), transforms.RandomRotation(degrees=3), transforms.CenterCrop(size=29), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])