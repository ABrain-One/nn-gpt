import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(29), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomSolarize(248, p=0.43), transforms.RandomAutocontrast(p=0.37), transforms.GaussianBlur(kernel_size=3, sigma=1.5, p=0.5), transforms.Pad(padding=4, fill=(7, 174, 130), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])