import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.RandomPosterize(bits=4, p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(**norm)])