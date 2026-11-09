import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(128), transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=torch.tensor([125.37 / 255.0, 122.88 / 255.0, 113.92 / 255.0]), std=torch.tensor([62.99 / 255.0, 62.08 / 255.0, 66.48 / 255.0]))])