import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(64, scale=(0.8, 1.2), ratio=(0.8, 2.0)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.4366547, 0.4462814, 0.4132817], std=[0.26743362, 0.26343278, 0.2788621])])