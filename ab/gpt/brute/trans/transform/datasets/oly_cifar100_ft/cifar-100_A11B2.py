import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.63), transforms.RandomPerspective(distortion_scale=0.22, p=0.67), transforms.RandomSolarize(threshold=192, p=0.23), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4974, 0.4817, 0.2801], std=[0.2802, 0.2712, 0.287])])