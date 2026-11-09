<code>
import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.8, 1.8)),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.5),<br>
&nbsp; &nbsp; &nbsp; transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),<br>
&nbsp; &nbsp; &nbsp; transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br>
&nbsp; &nbsp; ])
</code>