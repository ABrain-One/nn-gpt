import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=5, sigma=(0.33, 1.22)), transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(*norm)])