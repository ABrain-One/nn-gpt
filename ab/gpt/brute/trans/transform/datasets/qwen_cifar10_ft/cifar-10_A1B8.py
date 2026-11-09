import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(24, 73, 255), padding_mode='edge'), transforms.Grayscale(num_output_channels=3), transforms.RandomSolarize(threshold=204, p=0.46), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])