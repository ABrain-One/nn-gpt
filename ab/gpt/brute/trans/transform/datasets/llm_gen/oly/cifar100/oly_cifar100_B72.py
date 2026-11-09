import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(70, 155, 34), padding_mode='symmetric'), transforms.RandomPerspective(distortion_scale=0.23, p=0.8), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.CenterCrop(size=26), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])