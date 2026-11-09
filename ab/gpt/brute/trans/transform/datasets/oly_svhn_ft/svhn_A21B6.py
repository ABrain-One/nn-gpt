import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.73), transforms.GaussianBlur(kernel_size=5, sigma=(0.71, 1.24)), transforms.RandomSolarize(threshold=131, p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])