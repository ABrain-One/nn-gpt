import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(148, 89, 114), padding_mode='reflect'), transforms.RandomPerspective(distortion_scale=0.15, p=0.62), transforms.RandomSolarize(threshold=10, p=0.78), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])