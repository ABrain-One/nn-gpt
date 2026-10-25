import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(170, 213, 73), padding_mode='constant'),
    transforms.RandomSolarize(threshold=124, p=0.49),
    transforms.RandomPosterize(bits=8, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm))