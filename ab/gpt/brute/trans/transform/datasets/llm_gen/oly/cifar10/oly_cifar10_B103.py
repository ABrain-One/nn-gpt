import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(199, 77, 171), padding_mode='symmetric'), transforms.RandomRotation(degrees=17), transforms.RandomInvert(p=0.33), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(kernel_size=3, sigma=1.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])