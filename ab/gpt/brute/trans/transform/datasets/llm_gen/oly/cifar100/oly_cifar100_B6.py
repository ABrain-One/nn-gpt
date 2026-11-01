import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.75, 1.3333)), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2564, 0.2762])])