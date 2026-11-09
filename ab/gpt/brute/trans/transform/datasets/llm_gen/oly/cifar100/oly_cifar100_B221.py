import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(167, 252, 66), padding_mode='constant'), transforms.RandomCrop(32), transforms.RandomHorizontalFlip(p=0.68), transforms.RandomPerspective(distortion_scale=0.18, p=0.2), transforms.RandomEqualize(p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])