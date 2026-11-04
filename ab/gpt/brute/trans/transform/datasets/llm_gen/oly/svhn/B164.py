import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.8, 1.25), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.Pad(padding=4, fill=(0.485, 0.456, 0.406), padding_mode='constant'), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])