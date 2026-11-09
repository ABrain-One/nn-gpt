import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.Pad(padding=4, fill=(128, 124, 111), padding_mode='constant'), transforms.RandomAffine(degrees=10, translate=(-0.1, 0.1), scale=(0.95, 1.05), shear=2.0), transforms.GaussianBlur(kernel_size=5, sigma=(0.69, 1.46)), transforms.RandomHorizontalFlip(p=0.6), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4401], std=[0.2673, 0.2564, 0.2762])])