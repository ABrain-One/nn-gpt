import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.54, 0.94), ratio=(1.31, 2.03)), transforms.RandomPerspective(distortion_scale=0.15, p=0.6), transforms.RandomInvert(p=0.5), transforms.RandomGrayscale(p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.507, 0.486, 0.441], std=[0.267, 0.256, 0.276])])