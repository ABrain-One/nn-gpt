"""What each GA modification does to PSNR, measured from the fitness caches.

WHY THE CACHES. Nothing in the results database records which operator produced a model --
`nn.name` is a content hash and lineage is gone by the time a stat row is written. The caches keep
it: every entry carries `name`, `parent` (a NAME) and `psnr`, and a child's name is its parent's
plus one suffix token, so the operator is that trailing token (`+m` weighted mutation, `+s`
skip-operator edit, `X` infixed crossover).

A NAME IS NOT UNIQUE. One arm holds 682 entries under 44 names, because the name records the
operator *sequence* and never which mutation was drawn. Keying a dict by name keeps one arbitrary
row and drops ~94% of the sample.

UNIT OF ANALYSIS -- BOTH ARE REPORTED, BECAUSE THEY ANSWER DIFFERENT QUESTIONS.
  * EDGE level: one row per (parent node -> child node). ~306 rows. Asks "does applying this
    operator at this point in the lineage move the node's score", and is the cleaner causal unit.
  * CHILD level: one row per child MODEL, referenced to its parent group's mean. ~2,285 rows.
    Asks "what does a single draw of this operator do", which is the question about the operator
    itself. Children sharing a parent are NOT independent, so every interval here is a bootstrap
    over PARENT CLUSTERS, never over rows -- resampling rows would report a false precision,
    roughly sqrt(n_child/n_edge) ~ 2.7x too tight.

THREE OUTCOMES, because a single final PSNR is the noisiest thing the cache holds.
  * d_best  -- change in best-epoch PSNR. What the GA's fitness actually used.
  * d_avg   -- change in the mean over the epoch curve. Same quantity, lower variance, so this
               is the powered test; 52% of entries carry a full curve.
  * d_final -- change in last-epoch PSNR, kept only to show it agrees.
Plus `still_climbing`: whether the run was still improving when the 5-epoch screen cut it off. An
edit that changes the trajectory rather than the endpoint shows up here and nowhere else.

AND YIELD, which is not optional. Half of all children are `edge_reject` -- over the MAC cap or
unbuildable -- and never get a PSNR. An operator that mostly emits illegal models is not useful
however good its survivors look, so effect is always reported next to the fraction that survived.

METRIC CAVEAT. `psnr` is the 5-epoch centre-crop screening score, rank-blind at the frontier. It
is still the right metric here: it is what the search saw when it kept or dropped a child, so this
measures the operator's effect *on the search*. Convergence effects would need every child
retrained.

  python mutation_effect.py
"""
import argparse, glob, json, os, random, statistics as st, collections

def op_of(child, parent):
    if 'X' in child and not child.startswith(parent): return 'crossover'
    if not child.startswith(parent): return None
    return {'+m':'mutate', '+s':'skip-op'}.get(child[len(parent):])

def valid(r):
    return (not r.get('edge_reject')) and isinstance(r.get('psnr'), (int,float)) and r['psnr'] > 0

def es(r, key):
    e = r.get('epoch_stats')
    return e.get(key) if isinstance(e, dict) else None

def boot_ci(clusters, stat, iters=4000, seed=20260916):
    """Percentile CI resampling CLUSTERS (parent nodes), not rows."""
    rng = random.Random(seed)
    if len(clusters) < 2: return (float('nan'), float('nan'))
    out = []
    n = len(clusters)
    for _ in range(iters):
        pick = [clusters[rng.randrange(n)] for _ in range(n)]
        vals = [v for c in pick for v in c]
        if vals: out.append(stat(vals))
    out.sort()
    return (out[int(.025*len(out))], out[int(.975*len(out))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/shared/ssd/home/b-b-ahmed/thesis/sweep_out/mutation_effect.json')
    ap.add_argument('--iters', type=int, default=4000)
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ga_*_cache.json')))
    paths += sorted(glob.glob('/shared/ssd/home/b-b-ahmed/thesis/sweep_out/randwalk/rw_*_cache.json'))

    edges, kids = [], []
    yields = collections.defaultdict(lambda: [0,0])
    n_entries = 0
    for p in paths:
        try: d = json.load(open(p))
        except Exception: continue
        if not isinstance(d, dict): continue
        arm = os.path.basename(p).replace('ga_','').replace('_cache.json','').replace('rw_','randwalk-').replace('.json','')
        groups = collections.defaultdict(list)
        for v in d.values():
            if isinstance(v, dict) and v.get('name'):
                groups[v['name']].append(v); n_entries += 1
        for name, ch in groups.items():
            par = next((k.get('parent') for k in ch if k.get('parent')), None)
            if not par or par not in groups: continue
            op = op_of(name, par)
            if not op: continue
            yields[op][0] += sum(1 for k in ch if valid(k)); yields[op][1] += len(ch)
            pv = [k for k in groups[par] if valid(k)]
            cv = [k for k in ch if valid(k)]
            if not pv or not cv: continue
            node = '%s|%s' % (arm, par)
            pb = st.fmean([k['psnr'] for k in pv])
            pa = [es(k,'avg') for k in pv if es(k,'avg')]
            pf = [es(k,'final') for k in pv if es(k,'final')]
            psc= [es(k,'still_climbing') for k in pv if es(k,'still_climbing') is not None]
            for k in cv:
                row = dict(arm=arm, op=op, node=node, depth=par.count('+m')+par.count('+s'),
                           d_best=k['psnr']-pb)
                ka, kf = es(k,'avg'), es(k,'final')
                if ka is not None and pa: row['d_avg']  = ka-st.fmean(pa)
                if kf is not None and pf: row['d_final']= kf-st.fmean(pf)
                sc = es(k,'still_climbing')
                if sc is not None and psc:
                    row['d_climb'] = (1.0 if sc else 0.0) - st.fmean([1.0 if x else 0.0 for x in psc])
                kids.append(row)
            edges.append(dict(arm=arm, op=op, node=node,
                              depth=par.count('+m')+par.count('+s'),
                              d_best=st.fmean([k['psnr'] for k in cv])-pb))

    print('cache entries %d | lineage EDGES %d | CHILD models %d' % (n_entries, len(edges), len(kids)))
    print()
    print('=== YIELD: does the operator produce a legal model at all? ===')
    for op,(ok,tot) in sorted(yields.items(), key=lambda kv:-kv[1][1]):
        print('  %-10s %5d legal of %5d   %5.1f%% usable' % (op, ok, tot, 100*ok/tot))

    def report(rows, field, label, iters):
        vals = [(r['node'], r[field]) for r in rows if field in r]
        if len(vals) < 8: return
        cl = collections.defaultdict(list)
        for n,v in vals: cl[n].append(v)
        clusters = list(cl.values())
        flat = [v for _,v in vals]
        lo,hi = boot_ci(clusters, st.fmean, iters=iters)
        print('  %-10s n=%5d clust=%4d  mean %+6.3f [%+6.3f,%+6.3f] %-9s med %+6.3f  up %5.1f%%'
              % (label, len(flat), len(clusters), st.fmean(flat), lo, hi,
                 'RESOLVED' if (lo>0 or hi<0) else 'unresolved',
                 st.median(flat), 100*sum(1 for x in flat if x>0)/len(flat)))

    for field,title in (('d_avg','d_avg  (curve mean — the powered test)'),
                        ('d_best','d_best (what fitness used)'),
                        ('d_final','d_final')):
        print()
        print('=== %s — CHILD level, CI bootstrapped over parent clusters ===' % title)
        report(kids, field, 'all', a.iters)
        for op in ('mutate','skip-op'):
            report([r for r in kids if r['op']==op], field, op, a.iters)

    print()
    print('=== EDGE level (one row per lineage node), for comparison ===')
    report(edges, 'd_best', 'all', a.iters)
    for op in ('mutate','skip-op'):
        report([r for r in edges if r['op']==op], 'd_best', op, a.iters)

    print()
    print('=== TRAJECTORY: change in "still climbing at epoch 5" ===')
    for op in ('mutate','skip-op'):
        report([r for r in kids if r['op']==op], 'd_climb', op, a.iters)

    print()
    print('=== BY DEPTH (child level, d_avg) ===')
    for lo,hi in [(0,0),(1,2),(3,5),(6,10),(11,999)]:
        sel=[r for r in kids if lo<=r['depth']<=hi]
        report(sel,'d_avg','depth %d%s'%(lo,'' if lo==hi else '-%s'%(hi if hi<999 else '+')), a.iters)

    json.dump(dict(n_entries=n_entries, n_edges=len(edges), n_children=len(kids),
                   yields={k:v for k,v in yields.items()}, children=kids, edges=edges),
              open(a.out,'w'), indent=1)
    print('\nwrote ' + a.out)

if __name__ == '__main__': main()
