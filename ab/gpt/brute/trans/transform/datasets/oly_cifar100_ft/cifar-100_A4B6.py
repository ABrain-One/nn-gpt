import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.73), transforms.RandomSolarize(threshold=163, p=0.83), transforms.GaussianBlur(kernel_size=5, sigma=(0.53, 1.3)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])