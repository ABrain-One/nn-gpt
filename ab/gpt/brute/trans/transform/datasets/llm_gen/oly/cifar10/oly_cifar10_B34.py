<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    mean_r = round(norm[0][0] * 255)
    mean_g = round(norm[0][1] * 255)
    mean_b = round(norm[0][2] * 255)
    fill_color = (mean_r, mean_g, mean_b)
    
    return transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 0.9), ratio=(0.9, 1.1)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomApply([transforms.Posterize(5)], p=0.3),
        transforms.Pad(4, fill=fill_color, padding_mode='constant'),
        transforms.Resize(64),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm[0], std=norm[1]),
    ])
</pre>