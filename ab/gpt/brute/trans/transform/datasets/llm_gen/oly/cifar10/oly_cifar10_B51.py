<td colspan=2><pre>import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.2), interpolation=Image.BICUBIC),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=0.5),
        transforms.Resize(size=64, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</pre></td>