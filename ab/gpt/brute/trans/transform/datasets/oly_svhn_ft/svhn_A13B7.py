import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(160, 191, 143), padding_mode='reflect'), transforms.RandomSolarize(threshold=118, p=0.17), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])