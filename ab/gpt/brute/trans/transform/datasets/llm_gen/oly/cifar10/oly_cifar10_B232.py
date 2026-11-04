import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(125, 122, 111), padding_mode='reflect'), transforms.RandomEqualize(p=0.5), transforms.RandomPosterize(bits=5, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2475, 0.2435, 0.2262])])