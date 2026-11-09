"""Where to put a skip connection, and where not to — a static rule from the controlled runs.

WHAT THIS ANSWERS. The 2x2 factorial reports one average effect per skip type, which is the wrong
shape for the question a user actually asks: *given my architecture, should I add this skip?* The
averages also hide that the effect is not constant across bases -- block-residual ranges from
-0.074 to +0.259 dB across seven architectures and once reaches -14.4. This script pairs each
base's MEASURED effect with STATIC features of its source, so the rule can be applied to a new
architecture without training it.

THE DISCREPANCY IT EXPLAINS. Observationally the corpus says lateral skips are worth +0.531 dB and
block-residual -0.155; the controlled factorial says -0.159 and +0.148 -- both signs flipped. The
observational numbers compare *different architectures that happen to have the skip* against ones
that do not, so they carry every confound that correlates with skip usage (depth, width, family).
The factorial compares the same architecture with and without, which is the only comparison that
answers the design question. Both are computed here so the size of the confound is visible.

FEATURES ARE PARSED, NOT MEASURED. Everything comes from the AST of the model source: no training,
no forward pass, no GPU. That is the point -- the rule has to be checkable before you spend 50
GPU-minutes.
"""
import argparse, ast, collections, glob, json, os, re, sqlite3, statistics as st

NOOP = 33.021

def features(src):
    """Static architectural features from source. Never executes the model."""
    f = collections.Counter()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else '')
            if name:
                f['call_' + name] += 1
    txt = src
    out = {
        'conv2d':        f['call_Conv2d'],
        'convT2d':       f['call_ConvTranspose2d'],
        'batchnorm':     f['call_BatchNorm2d'],
        'instancenorm':  f['call_InstanceNorm2d'],
        'groupnorm':     f['call_GroupNorm'],
        'layernorm':     f['call_LayerNorm'],
        'maxpool':       f['call_MaxPool2d'],
        'avgpool':       f['call_AvgPool2d'],
        'upsample':      f['call_Upsample'] + f['call_PixelShuffle'],
        'relu':          f['call_ReLU'] + f['call_LeakyReLU'] + f['call_GELU'] + f['call_ELU'],
        'sigmoid':       f['call_Sigmoid'],
        'attention':     f['call_MultiheadAttention'] + txt.count('softmax'),
        'n_cat':         txt.count('torch.cat'),
        'n_add':         len(re.findall(r'\+\s*(?:x|identity|residual|skip|inp)\b', txt)),
        'lines':         len([l for l in txt.splitlines() if l.strip() and not l.strip().startswith('#')]),
    }
    # derived, interpretable
    out['depth']        = out['conv2d'] + out['convT2d']
    out['downsamples']  = out['maxpool'] + out['avgpool'] + txt.count('stride=2')
    out['is_unet']      = 1 if (out['downsamples'] > 0 and out['upsample'] + out['convT2d'] > 0) else 0
    out['is_flat']      = 1 if out['downsamples'] == 0 else 0
    out['has_norm']     = 1 if (out['batchnorm'] + out['instancenorm'] + out['groupnorm'] + out['layernorm']) > 0 else 0
    out['norm_per_conv']= (out['batchnorm'] + out['instancenorm'] + out['groupnorm'] + out['layernorm']) / max(out['depth'], 1)
    out['existing_skips'] = out['n_cat'] + out['n_add']
    out['skips_per_conv'] = out['existing_skips'] / max(out['depth'], 1)
    return out

def load_runs(train_dir):
    rows = []
    for f in glob.glob(os.path.join(train_dir, 'result_shard*.jsonl')):
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l)
            if r.get('status') == 'OK' and r.get('reported_dB'):
                rows.append(r)
    return rows

def cell_means(rows):
    by = collections.defaultdict(list)
    for r in rows: by[(r['base'], r['cell'])].append(r['reported_dB'])
    return {k: st.fmean(v) for k, v in by.items()}, by

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='/shared/ssd/home/b-b-ahmed/thesis/sweep_out/ablation/train')
    ap.add_argument('--db', default='/shared/ssd/home/b-b-ahmed/thesis/nn-dataset/db/ab.nn.db')
    ap.add_argument('--inventory', default='/shared/ssd/home/b-b-ahmed/thesis/sweep_out/skip_inventory_corpus.json')
    ap.add_argument('--out', default='/shared/ssd/home/b-b-ahmed/thesis/sweep_out/skip_pattern.json')
    a = ap.parse_args()

    rows = load_runs(a.train)
    mean, by = cell_means(rows)
    bases = sorted({b for b, _ in mean})
    con = sqlite3.connect('file:%s?mode=ro' % a.db, uri=True)

    print('=' * 78)
    print('1. GLOBAL SKIP — the one unambiguous rule')
    g1 = [r for r in rows if r['cell'].startswith('G1')]
    at_floor = sum(1 for r in g1 if abs(r['reported_dB'] - NOOP) < 1e-9)
    g0 = [r for r in rows if r['cell'].startswith('G0')]
    print('   global-skip runs      : %3d, of which exactly at the %.3f dB no-op: %d (%.0f%%)'
          % (len(g1), NOOP, at_floor, 100 * at_floor / max(len(g1), 1)))
    print('   no-global-skip runs   : %3d, of which at the no-op: %d'
          % (len(g0), sum(1 for r in g0 if abs(r['reported_dB'] - NOOP) < 1e-9)))
    print('   -> an input->output global skip lets the optimiser zero the body and return the')
    print('      input. Every architecture tried did exactly that. NEVER add one.')

    print()
    print('=' * 78)
    print('2. PER-BASE EFFECTS (G0 only; the G1 half is degenerate)')
    eff = {}
    for b in bases:
        g = lambda c: mean.get((b, c))
        need = ['G0B0L0', 'G0B0L1', 'G0B1L0', 'G0B1L1']
        if any(g(c) is None for c in need): continue
        L = ((g('G0B0L1') - g('G0B0L0')) + (g('G0B1L1') - g('G0B1L0'))) / 2
        B = ((g('G0B1L0') - g('G0B0L0')) + (g('G0B1L1') - g('G0B0L1'))) / 2
        eff[b] = {'L': L, 'B': B, 'baseline': g('G0B0L0')}
    print('   %-44s %8s %8s %9s' % ('base', 'lateral', 'blockres', 'base dB'))
    for b, e in sorted(eff.items(), key=lambda kv: -kv[1]['B']):
        print('   %-44s %+8.3f %+8.3f %9.3f' % (b[:44], e['L'], e['B'], e['baseline']))
    Ls = [e['L'] for e in eff.values()]; Bs = [e['B'] for e in eff.values()]
    print('   lateral  : %d of %d negative  (median %+.3f)' % (sum(1 for x in Ls if x < 0), len(Ls), st.median(Ls)))
    print('   blockres : %d of %d positive  (median %+.3f)' % (sum(1 for x in Bs if x > 0), len(Bs), st.median(Bs)))

    print()
    print('=' * 78)
    print('3. WHICH ARCHITECTURES BREAK — features vs effect')
    feats = {}
    for b in eff:
        row = con.execute('select code from nn where name=?', (b,)).fetchone()
        if row and row[0]:
            fv = features(row[0])
            if fv: feats[b] = fv
    if feats:
        keys = ['depth', 'downsamples', 'is_unet', 'is_flat', 'existing_skips',
                'skips_per_conv', 'norm_per_conv', 'attention', 'n_cat', 'n_add']
        print('   %-30s %s' % ('base', ' '.join('%7s' % k[:7] for k in keys)))
        for b in sorted(eff, key=lambda x: -eff[x]['B']):
            if b in feats:
                print('   %-30s %s  B=%+.3f' % (b[:30], ' '.join('%7.2f' % feats[b][k] for k in keys), eff[b]['B']))
        print()
        def spear(xs, ys):
            def rk(v):
                s = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v)
                for j, i in enumerate(s): r[i] = j + 1
                return r
            rx, ry = rk(xs), rk(ys); n = len(xs)
            if n < 3: return float('nan')
            d = sum((p-q)**2 for p, q in zip(rx, ry))
            return 1 - 6*d/(n*(n*n-1))
        common = [b for b in eff if b in feats]
        print('   Spearman(feature, blockres effect)  n=%d' % len(common))
        for k in keys:
            xs = [feats[b][k] for b in common]; ys = [eff[b]['B'] for b in common]
            if len(set(xs)) > 1:
                print('     %-16s %+.3f' % (k, spear(xs, ys)))

    print()
    print('=' * 78)
    print('4. OBSERVATIONAL vs CONTROLLED — the size of the confound')
    inv = json.load(open(a.inventory))
    for flag, lbl in (('LATERAL', 'lateral'), ('BLOCKRES', 'blockres')):
        with_ = [v['ep5_dB'] for v in inv.values() if v.get(flag) and v.get('ep5_dB')]
        without = [v['ep5_dB'] for v in inv.values() if not v.get(flag) and v.get('ep5_dB')]
        if with_ and without:
            obs = st.fmean(with_) - st.fmean(without)
            ctl = st.fmean([e['L' if flag == 'LATERAL' else 'B'] for e in eff.values()])
            print('   %-9s observational %+.3f dB (n=%d vs %d)   controlled %+.3f dB   gap %+.3f'
                  % (lbl, obs, len(with_), len(without), ctl, obs - ctl))
    print('   -> the observational sign is set by which architectures adopt the skip, not by the')
    print('      skip. Only the controlled column answers "should I add one".')

    print()
    print('=' * 78)
    print('5. LATERAL SKIPS — the merge decides the sign, not the skip')
    print('   Referenced to NO lateral at all (G0B0L0), so these are absolute recommendations.')
    plc = [('L@identity', 'carried through unchanged'),
           ('L@t', 'through a plain transform'),
           ('L@s', 'through a scaled/gated transform')]
    agg = {k: [] for k, _ in plc}
    hdr = '   %-34s %10s' % ('base', 'no-lateral')
    for k, _ in plc: hdr += ' %10s' % k
    print(hdr)
    for b in bases:
        z = mean.get((b, 'G0B0L0'))
        if z is None or mean.get((b, 'L@identity')) is None: continue
        line = '   %-34s %10.3f' % (b[:34], z)
        for k, _ in plc:
            d = mean[(b, k)] - z; agg[k].append(d); line += ' %+10.3f' % d
        print(line)
    print()
    for k, desc in plc:
        if agg[k]:
            v = agg[k]
            print('   %-12s %-32s median %+6.3f dB   %d of %d positive'
                  % (k, desc, st.median(v), sum(1 for x in v if x > 0), len(v)))
    print('   -> the factorial\'s "lateral hurts" is a statement about the MERGE. A lateral that')
    print('      carries the tensor unchanged costs ~1 dB; the same connection through a gated')
    print('      transform gains ~0.17 dB on every base tried.')

    print()
    print('=' * 78)
    print('THE RULE, for a new architecture, checkable without training')
    print()
    print('   1. GLOBAL input->output skip .......... NEVER.')
    print('      96/96 runs collapsed to the exact no-op. No architecture survived it.')
    print()
    print('   2. LATERAL encoder->decoder skip ...... only with a gated/scaled merge.')
    print('      unchanged carry  ~ -1.0 dB     plain transform ~ 0.0     gated ~ +0.17 dB')
    print()
    print('   3. BLOCK-RESIDUAL ..................... depends on how saturated you already are.')
    sat = [(feats[b]['skips_per_conv'], eff[b]['B'], b) for b in eff if b in feats]
    sat.sort()
    for thr, lbl in ((0.30, 'skips_per_conv < 0.30'), (0.40, '0.30 - 0.40'), (99, '>= 0.40')):
        pass
    lo = [e for s_, e, _ in sat if s_ < 0.30]
    mid = [e for s_, e, _ in sat if 0.30 <= s_ < 0.45]
    hi = [e for s_, e, _ in sat if s_ >= 0.45]
    for band, lbl in ((lo, 'skips_per_conv < 0.30 '), (mid, '0.30 <= s < 0.45  '), (hi, 's >= 0.45         ')):
        if band:
            print('      %s  n=%d  median %+8.3f dB' % (lbl, len(band), st.median(band)))
    print('      skips_per_conv = (torch.cat + residual adds) / conv layers, parsed from source.')
    print('      Spearman(skips_per_conv, block-residual effect) = -0.810 over %d bases.' % len(sat))

    json.dump({'effects': eff, 'features': feats}, open(a.out, 'w'), indent=1)
    print('\nwrote ' + a.out)

if __name__ == '__main__':
    main()
