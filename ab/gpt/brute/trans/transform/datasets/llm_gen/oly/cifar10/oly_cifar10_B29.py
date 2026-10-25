<td colspan="2">
<pre><code>&lt;tr&gt;
import torch&lt;br /&gt;
from torchvision import transforms&lt;br /&gt;
&nbsp;&lt;br /&gt;
def transform(norm):&lt;br /&gt;
 \treturn transforms.Compose([&lt;br /&gt;
 \t\ttransforms.Resize(224),&lt;br /&gt;
 \t\ttransforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.333)),&lt;br /&gt;
 \t\ttransforms.RandomHorizontalFlip(p=0.5),&lt;br /&gt;
 \t\ttransforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),&lt;br /&gt;
 \t\ttransforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),&lt;br /&gt;
 \t\ttransforms.ToTensor(),&lt;br /&gt;
 \t\ttransforms.Normalize(**norm)&lt;br /&gt;
 \t]) &lt;/tr&gt;
</code></pre>
</td>