import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.22, p=0.48), transforms.RandomEqualize(p=0.59), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])