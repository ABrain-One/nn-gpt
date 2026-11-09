<td>
<code>
<pre>
import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.CenterCrop(26),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomHorizontalFlip(p=0.9),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomAdjustSharpness(1.93, p=0.86),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br>
&nbsp;&nbsp;])<br>
</pre>
</code>
</td>