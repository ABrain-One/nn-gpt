<pre>
import torch
from torchvision import transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=1.5)], p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.486, 0.441], std=[0.267, 0.256, 0.276])
    ])
</pre>