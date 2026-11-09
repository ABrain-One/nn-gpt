import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Pad(padding=1, fill=(118, 192, 105), padding_mode='constant'), transforms.RandomAutocontrast(p=0.8), transforms.RandomSolarize(threshold=152, p=0.6), transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(p=0.5), transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2673, 0.2565, 0.2531])])