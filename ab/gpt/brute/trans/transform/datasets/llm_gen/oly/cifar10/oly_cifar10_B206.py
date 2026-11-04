import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomVerticalFlip(p=0.78), T.RandomHorizontalFlip(p=0.13), T.RandomApply([T.Solarize(171)], p=0.3), T.RandomGrayscale(p=0.2), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1.5)], p=0.5), T.Resize((64, 64)), T.ToTensor(), T.Normalize(**norm)])