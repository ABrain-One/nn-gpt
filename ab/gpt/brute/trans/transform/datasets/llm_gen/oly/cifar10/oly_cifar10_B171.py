import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize((96, 96)), transforms.RandomCrop((64, 64)), transforms.RandomRotation(15), transforms.RandomHorizontalFlip(p=0.5), transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), transforms.ToTensor(), transforms.Normalize([0.49139973, 0.48215845, 0.4465309], [0.24703225, 0.24348292, 0.2615844])])