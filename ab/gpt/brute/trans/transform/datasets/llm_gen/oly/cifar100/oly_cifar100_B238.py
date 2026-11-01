import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.85), transforms.RandomCrop(24), transforms.Pad(padding=3, fill=(117, 233, 66), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762])])