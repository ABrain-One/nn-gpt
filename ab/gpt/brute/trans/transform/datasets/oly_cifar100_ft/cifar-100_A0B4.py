import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(4, padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.3, p=0.5), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomAdjustSharpness(sharpness_factor=0.5, p=0.5), transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.486, 0.458], std=[0.267, 0.256, 0.276])])