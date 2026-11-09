import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomGrayscale(p=0.2), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=1.0)], p=0.5), transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.3), padding=8, fill=[int(255 * m) for m in norm[0]]), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.ToTensor(), transforms.Normalize(*norm)])