import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=25), transforms.GaussianBlur(kernel_size=3, sigma=(0.36, 1.18)), transforms.RandomSolarize(threshold=163, p=0.45), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])