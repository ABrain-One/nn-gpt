import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(104, 102, 110), padding_mode='constant'), transforms.RandomSolarize(threshold=221, p=0.16), transforms.RandomVerticalFlip(p=0.52), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])