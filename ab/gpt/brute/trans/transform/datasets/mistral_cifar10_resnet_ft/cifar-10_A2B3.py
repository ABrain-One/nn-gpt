import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=26), transforms.RandomEqualize(p=0.9), transforms.RandomAdjustSharpness(sharpness_factor=1.98, p=0.86), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])