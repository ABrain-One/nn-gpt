import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
&nbsp;&nbsp;return transforms.Compose([<br/>
&thinsp;transforms.RandomResizedCrop(64, scale=(0.5, 2.0), ratio=(0.8, 1.2)),<br/>
&thinsp;transforms.RandomHorizontalFlip(p=0.5),<br/>
&thinsp;transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),<br/>
&thinsp;transforms.RandomGrayscale(p=0.25),<br/>
&thinsp;transforms.GaussianBlur(kernel_size=3, sigma=1.0),<br/>
&thinsp;transforms.ToTensor(),<br/>
&thinsp;transforms.Normalize(**norm)<br/>
&thinsp;])<br/>