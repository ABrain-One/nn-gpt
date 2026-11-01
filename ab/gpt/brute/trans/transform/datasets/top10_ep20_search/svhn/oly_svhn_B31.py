import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.2), ratio=(0.75, 1.333)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4385, 0.444, 0.4731], std=[0.2616, 0.2574, 0.2539])])