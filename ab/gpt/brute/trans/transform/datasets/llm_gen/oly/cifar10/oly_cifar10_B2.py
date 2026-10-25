import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(244, 255, 133), padding_mode='symmetric'), transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomApply([transforms.ColorJitter(hue=(-0.1, 0.1), saturation=(0.8, 1.2))], p=0.8), transforms.RandomRotation(degrees=15), transforms.GaussianBlur(kernel_size=3, sigma=0.5), transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])