import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(242, 250, 177), padding_mode='edge'), transforms.RandomSolarize(threshold=143, p=0.42), transforms.RandomGrayscale(p=0.49), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])