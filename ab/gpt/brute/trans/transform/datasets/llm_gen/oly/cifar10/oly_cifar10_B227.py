import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.76), transforms.RandomCrop(30), transforms.RandomSolarize(threshold=37, p=0.45), transforms.RandomAdjustSharpness(0.54, p=0.28), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])