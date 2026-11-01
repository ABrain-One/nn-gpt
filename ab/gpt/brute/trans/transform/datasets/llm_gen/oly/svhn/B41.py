import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(231, 98, 54), padding_mode='symmetric'), transforms.ColorJitter(brightness=0.9, contrast=0.95, saturation=1.1, hue=0.05), transforms.RandomSolarize(threshold=105, p=0.8), transforms.RandomVerticalFlip(p=0.5), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])