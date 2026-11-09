import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
&nbsp; return transforms.Compose([<br/>
&nbsp;&nbsp; transforms.RandomHorizontalFlip(p=0.7),<br/>
&nbsp;&nbsp; transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),<br/>
&nbsp;&nbsp; transforms.GaussianBlur(kernel_size=3, sigma=1.0),<br/>
&nbsp;&nbsp; transforms.Pad(padding=4, fill=(19, 120, 141), padding_mode='constant'),<br/>
&nbsp;&nbsp; transforms.Resize((64, 64)),<br/>
&nbsp;&nbsp; transforms.ToTensor(),<br/>
&nbsp;&nbsp; transforms.Normalize(*norm)<br/>
&nbsp; ])