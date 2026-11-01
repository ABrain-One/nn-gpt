import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.RandomAdjustSharpness(sharpness_factor=1.96, p=0.83), transforms.RandomSolarize(threshold=161, p=0.33), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])