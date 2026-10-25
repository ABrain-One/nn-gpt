import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomSolarize(threshold=16, p=0.44), transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])