import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(1, fill=(126, 124, 95), padding_mode='constant'), transforms.RandomEqualize(p=0.5), transforms.RandomRotation(degrees=(-7, 7), p=0.7), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6), transforms.RandomGrayscale(p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=norm[0], std=norm[1])])