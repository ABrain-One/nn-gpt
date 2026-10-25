import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Resize(64), transforms.CenterCrop(31), transforms.RandomAffine(degrees=15, translate=(0.08, 0.03), scale=(0.94, 1.96), shear=(4.16, 8.05)), transforms.ColorJitter(brightness=1.16, contrast=0.92, saturation=1.05, hue=0.04), transforms.RandomGrayscale(p=0.12), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4823, 0.4465], std=[0.247, 0.2435, 0.2616])])