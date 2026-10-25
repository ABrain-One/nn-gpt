import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(200, 32, 165), padding_mode='constant'), transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(1.0, 1.0)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomRotation(5), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.RandomGrayscale(p=0.05), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])