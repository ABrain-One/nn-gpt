<td>
<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.8, contrast=1.0, saturation=0.8, hue=0.1),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.5),
        transforms.RandomSolarize(threshold=170, p=0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</pre>
</td>