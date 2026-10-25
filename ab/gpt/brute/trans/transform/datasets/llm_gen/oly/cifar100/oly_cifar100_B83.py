<td colspan="2">
<div class="codehilite"><pre><span></span><code><span class="kn">import</span> <span class="nn">torch</span>
<span class="kn">import</span> <span class="nn">torchvision.transforms</span> <span class="k">as</span> <span class="n">F</span>

<span class="k">def</span> <span class="nf">transform</span><span class="p">():</span>
    <span class="k">return</span> F.<span class="n">Compose</span><span class="p">([</span>
        F.<span class="n">RandomResizedCrop</span><span class="p">(</span><span class="mi">224</span>, <span class="n">scale</span><span class="o">=</span><span class="p">(</span><span class="mf">0.8</span>, <span class="mf">2.0</span><span class="p">),</span> <span class="n">ratio</span><span class="o">=</span><span class="p">(</span><span class="mf">0.9</span>, <span class="mf">1.1</span><span class="p">)),</span>
        F.<span class="n">RandomHorizontalFlip</span><span class="p">(</span><span class="n">p</span><span class="o">=</span><span class="mf">0.5</span><span class="p">),</span>
        F.<span class="n">ColorJitter</span><span class="p">(</span><span class="n">brightness</span><span class="o">=</span><span class="mf">0.8</span>, <span class="n">contrast</span><span class="o">=</span><span class="mf">0.9</span>, <span class="n">saturation</span><span class="o">=</span><span class="mf">1.0</span>, <span class="n">hue</span><span class="o">=</span><span class="mf">0.1</span><span class="p">),</span>
        F.<span class="n">GaussianBlur</span><span class="p">(</span><span class="n">kernel_size</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">sigma</span><span class="o">=</span><span class="mf">1.0</span><span class="p">),</span>
        F.<span class="n">ToTensor</span><span class="p">(),</span>
        F.<span class="n">Normalize</span><span class="p">(</span>
            <span class="n">mean</span><span class="o">=</span><span class="p">[[</span><span class="mf">0.4914</span><span class="p">,</span> <span class="mf">0.4822</span><span class="p">,</span> <span class="mf">0.4465</span><span class="p">]],</span>
            <span class="n">std</span><span class="o">=</span><span class="p">[[</span><span class="mf">0.2023</span><span class="p">,</span> <span class="mf">0.1994</span><span class="p">,</span> <span class="mf">0.2010</span><span class="p">]]</span>
        <span class="p">)</span>
    <span class="p">])</span>
</code></pre>
</div>
</td>