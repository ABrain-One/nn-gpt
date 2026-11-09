import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(148, 75, 24), padding_mode='edge'), transforms.RandomAdjustSharpness(sharpness_factor=0.73, p=0.52), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])