import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5.0, 5.0), resample=transforms.InterpolationMode.BICUBIC, fill=(132, 123, 112)), transforms.GaussianBlur(kernel_size=5, sigma=1.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.5071, 0.4867, 0.4401], std=[0.2673, 0.2564, 0.2762])])