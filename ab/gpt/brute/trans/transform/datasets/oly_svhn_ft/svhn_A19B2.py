import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(142, 144, 174), padding_mode='edge'), transforms.RandomSolarize(threshold=189, p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])