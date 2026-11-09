import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(184, 21, 193), padding_mode='edge'), transforms.RandomSolarize(threshold=184, p=0.64), transforms.RandomAutocontrast(p=0.67), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])