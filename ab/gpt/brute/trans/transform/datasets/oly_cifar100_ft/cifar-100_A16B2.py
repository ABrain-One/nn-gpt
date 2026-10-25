import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=28), transforms.RandomRotation(degrees=16), transforms.RandomVerticalFlip(p=0.18), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.481, 0.461, 0.721], std=[0.251, 0.271, 0.771])])