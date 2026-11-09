import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.486, 0.441], std=[0.267, 0.256, 0.276])])