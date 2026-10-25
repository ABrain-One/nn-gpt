<code>
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

def transform(norm):
    rand_apply_prob = 0.5
    color_jitter_prob = 0.5
    solarize_prob = 0.25
    
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))], p=rand_apply_prob),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=color_jitter_prob),
        transforms.RandomApply([transforms.Solarize(threshold=153)], p=solarize_prob),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.7070], std=[0.2023, 0.1994, 0.1175])
    ])
</code>