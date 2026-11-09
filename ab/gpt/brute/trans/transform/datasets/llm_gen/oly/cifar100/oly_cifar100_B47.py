import torch<br/>
import torchvision.transforms as transforms<br/>
&nbsp;<br/>
def transform(norm):<br/>
 &nbsp;return transforms.Compose([<br/>
 &nbsp; transforms.RandomResizedCrop(32, scale=(0.6, 1.2), ratio=(0.9, 1.1)),<br/>
 &nbsp; transforms.RandomRotation(15),<br/>
 &nbsp; transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),<br/>
 &nbsp; transforms.GaussianBlur(kernel_size=3, sigma=1.5),<br/>
 &nbsp; transforms.Resize((64, 64)),<br/>
 &nbsp; transforms.ToTensor(),<br/>
 &nbsp; transforms.Normalize(*norm)<br/>
 &nbsp;])<br/>