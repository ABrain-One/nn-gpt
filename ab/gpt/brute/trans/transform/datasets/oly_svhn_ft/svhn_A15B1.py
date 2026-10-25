import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=25), transforms.RandomSolarize(threshold=36, p=0.35), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])