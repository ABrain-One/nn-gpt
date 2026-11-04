<td colspan="3"><pre style="white-space: pre-wrap; font-family:monospace;">import torch<br />
import torchvision.transforms as transforms<br />
<br />
def transform(norm):<br />
&nbsp; return transforms.Compose([<br />
&nbsp;&nbsp; transforms.RandomResizedCrop(32, scale=(0.75, 1.25), ratio=(0.9, 1.1)),<br />
&nbsp;&nbsp; transforms.RandomHorizontalFlip(p=0.5),<br />
&nbsp;&nbsp; transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br />
&nbsp;&nbsp; transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=0.5)], p=0.5),<br />
&nbsp;&nbsp; transforms.ToTensor(),<br />
&nbsp;&nbsp; transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])<br />
&nbsp; ])</pre></td>