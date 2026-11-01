import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(159, 128, 94), padding_mode='constant'), transforms.RandomCrop(size=31), transforms.Grayscale(num_output_channels=3), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])