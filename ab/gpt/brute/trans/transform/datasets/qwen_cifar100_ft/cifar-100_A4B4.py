import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.Pad(padding=[1, 2, 2, 1], fill=(233, 100, 249), padding_mode='reflect'), transforms.RandomSolarize(threshold=146, p=0.2), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])