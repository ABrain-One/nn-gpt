import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomCrop(30), transforms.RandomPerspective(distortion_scale=0.12, p=0.4), transforms.GaussianBlur(kernel_size=5, sigma=(0.82, 1.49), p=0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(**norm)])