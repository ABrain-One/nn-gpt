<td>
<pre><code>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.64, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(**norm)
    ])
</code></pre>
</td>