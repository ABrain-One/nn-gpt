<td>
<code>
<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.6, 1.0), ratio=(0.8, 2.0)),
        transforms.RandomApply([transforms.Equalize()], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
</pre>
</code>
</td>