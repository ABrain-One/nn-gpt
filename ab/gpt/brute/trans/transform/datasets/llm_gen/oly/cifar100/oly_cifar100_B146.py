import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(25), transforms.ColorJitter(brightness=1.15, contrast=0.9, saturation=1.0, hue=0.03), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomGrayscale(p=0.2), transforms.RandomPosterize(5, p=0.7), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])