import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(31, 112, 50), padding_mode='edge'), transforms.RandomHorizontalFlip(p=0.63), transforms.RandomVerticalFlip(p=0.83), transforms.RandomSolarize(threshold=90, p=0.46), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])