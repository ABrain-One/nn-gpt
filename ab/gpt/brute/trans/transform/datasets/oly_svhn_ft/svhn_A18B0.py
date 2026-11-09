import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(161, 21, 105), padding_mode='symmetric'), transforms.RandomSolarize(threshold=164, p=0.83), transforms.RandomPosterize(bits=8, p=0.55), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])