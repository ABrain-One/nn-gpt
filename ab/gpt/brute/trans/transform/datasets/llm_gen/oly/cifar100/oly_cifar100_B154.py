import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.73), transforms.RandomVerticalFlip(p=0.82), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomRotation(degrees=10), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.424], std=[0.2673, 0.2565, 0.2382])])