import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5), transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=1.16, hue=0.07), transforms.RandomAdjustSharpness(1.45, p=0.5), transforms.RandomSolarize(threshold=207, p=0.5), transforms.RandomRotation(degrees=(-5, 5), p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])