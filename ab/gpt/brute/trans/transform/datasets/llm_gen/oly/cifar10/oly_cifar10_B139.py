import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAffine(degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5)), transforms.RandomHorizontalFlip(p=0.8), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Solarize(threshold=200, p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.2435, 0.2616])])