import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 1.2)), T.RandomRotation(15), T.RandomPerspective(distortion_scale=0.2, p=0.2), T.GaussianBlur(kernel_size=3, sigma=0.5), T.ColorJitter(hue=0.1, brightness=0.2, contrast=0.2, saturation=0.2), T.ToTensor(), T.Normalize(*norm)])