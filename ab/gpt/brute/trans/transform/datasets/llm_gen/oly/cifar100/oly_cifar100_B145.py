import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomResizedCrop(32, scale=(0.8, 1.25), interpolation=T.Interpolation.BICUBIC), T.ColorJitter(0.1, 0.1, 0.1, 0.1), T.RandomApply([T.GaussianBlur(3)], p=0.5), T.Resize(64), T.ToTensor(), T.Normalize(*norm)])