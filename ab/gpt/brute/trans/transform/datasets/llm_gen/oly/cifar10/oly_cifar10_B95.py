import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(160, 71, 110), padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.2, p=0.2), transforms.RandomHorizontalFlip(p=0.1), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])