<td>
        <pre>
import torch
import torchvision.transforms as transforms

def transform():
    return transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139975, 0.48215856, 0.44653028], std=[0.2023, 0.1994, 0.2010])
    ])
</pre>
    </td>