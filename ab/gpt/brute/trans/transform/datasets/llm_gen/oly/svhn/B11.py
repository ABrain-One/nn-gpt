import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(0.89, 0.87, 0.88, 0.06), transforms.RandomAffine(5, (0.17, 0.16), (0.98, 1.55), (3.32, 6.07)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])