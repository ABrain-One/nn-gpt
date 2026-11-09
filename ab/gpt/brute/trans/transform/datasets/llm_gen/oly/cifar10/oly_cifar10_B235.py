import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 1.5))], p=0.5), transforms.ColorJitter(hue=0.1, saturation=0.3, brightness=0.2, contrast=0.3), transforms.RandomAdjustSharpness(sharpness_factor=(0.5, 2.0), p=0.4), transforms.Pad(padding=4, fill=(79, 251, 155), padding_mode='reflect'), transforms.RandomResizedCrop(32, scale=(0.8, 1.0)), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])