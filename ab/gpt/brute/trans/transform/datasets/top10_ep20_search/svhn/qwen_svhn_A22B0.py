import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomSolarize(threshold=170, p=0.39), transforms.RandomEqualize(p=0.31), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])