import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomAutocontrast(p=0.55), transforms.RandomPerspective(distortion_scale=0.26, p=0.44), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])