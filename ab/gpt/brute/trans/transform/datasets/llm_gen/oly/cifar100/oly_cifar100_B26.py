<td colspan="3"><pre><code>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomVerticalFlip(p=0.76),
        transforms.GaussianBlur(kernel_size=3, sigma=1.0),
        transforms.RandomAdjustSharpness(sharpness_factor=1.16, p=0.9),
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
</code></pre></td>