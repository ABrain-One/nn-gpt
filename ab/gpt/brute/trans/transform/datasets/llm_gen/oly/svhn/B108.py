import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.CenterCrop(28), transforms.RandomPosterize(bits=5, p=0.5), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])