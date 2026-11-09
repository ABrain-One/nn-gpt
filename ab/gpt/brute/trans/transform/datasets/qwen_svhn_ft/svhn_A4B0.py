import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.RandomAutocontrast(p=0.89), transforms.RandomPerspective(distortion_scale=0.27, p=0.32), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])