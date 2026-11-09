import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomResizedCrop(size=32, scale=(0.76, 0.95), ratio=(1.17, 1.79)), transforms.RandomEqualize(p=0.63), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])