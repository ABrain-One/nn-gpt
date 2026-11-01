import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224, scale=(0.75, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2531])])