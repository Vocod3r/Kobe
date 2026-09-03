"""Compare the original failed TD3 efficiency sweep with the real-SAC sweep.

Reads:
  tests/results/slider_sweep_efficiency.json   (TD3, original failed sweep)
  tests/results/sac_efficiency_sweep.json      (SAC, this run)

and prints side-by-side summary table, Pearson (efficiency -> speed), the
within-seed de-meaned correlation (Metric-2 style), and the per-seed
"seed-locked plateau" check (identical eval triples across slider values).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TD3_PATH = ROOT / 'tests' / 'results' / 'slider_sweep_efficiency.json'
SAC_PATH = ROOT / 'tests' / 'results' / 'sac_efficiency_sweep.json'


def load(path):
    data = json.loads(Path(path).read_text())
    return data['metadata'], data['runs']


def table(runs, key='efficiency_slider'):
    rows = {}
    for r in runs:
        rows[(r[key], r['seed'])] = r
    return rows


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan'), float('nan')
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def within_seed_stats(rows):
    """Return (overall_r, overall_p, within_r, list of per-seed r)."""
    xs = np.array([k[0] for k in rows])
    ys = np.array([rows[k]['speed'] for k in rows])
    r_over, p_over = pearson(xs, ys)
    seeds = sorted({k[1] for k in rows})
    per_seed = []
    demean = ys.astype(float).copy()
    for s in seeds:
        idx = [i for i, k in enumerate(rows) if k[1] == s]
        ms = float(np.mean([ys[i] for i in idx]))
        per_seed.append((s, pearson([xs[i] for i in idx], [ys[i] for i in idx])))
        for i in idx:
            demean[i] -= ms
    r_within, p_within = pearson(xs, demean)
    return r_over, p_over, r_within, per_seed


def plateaus(rows, key='efficiency_slider'):
    """Exact-repetition plateaus: same (speed,safety,comfort) triple at >=2 sliders."""
    found = []
    for s in sorted({r['seed'] for r in rows}):
        by_triple = {}
        for r in rows:
            if r['seed'] != s:
                continue
            trip = (round(r['speed'], 1), r['safety'], r['convenience'])
            by_triple.setdefault(trip, []).append(r[key])
        for trip, vals in sorted(by_triple.items(), key=lambda kv: -len(kv[1])):
            if len(vals) >= 2:
                found.append((s, trip, sorted(vals)))
    return found


def main() -> int:
    td3_meta, td3_runs = load(TD3_PATH)
    sac_meta, sac_runs = load(SAC_PATH)
    td3 = table(td3_runs)
    sac = table(sac_runs)
    values = sorted({k[0] for k in td3} | {k[0] for k in sac})
    seeds = sorted({k[1] for k in td3} | {k[1] for k in sac})

    print('=' * 94)
    print('STEP 7 - efficiency sweep: TD3 (original failed) vs real SAC')
    print(f"program: {sac_meta.get('program', 'n/a')}  | steps/run 5000 | seeds {seeds}")
    print('=' * 94)
    hdr = f"{'eff':>5} {'seed':>6} {'TD3 speed':>12} {'SAC speed':>12} {'d(SAC-TD3)':>12}"
    print(hdr)
    print('-' * 94)
    for v in values:
        for s in seeds:
            a = td3.get((v, s), {}).get('speed')
            b = sac.get((v, s), {}).get('speed')
            if a is None and b is None:
                continue
            d = (b - a) if (a is not None and b is not None) else float('nan')
            print(f"{v:>5.1f} {s:>6} {a if a is not None else float('nan'):>12.1f} "
                  f"{b if b is not None else float('nan'):>12.1f} {d:>12.1f}")
    print('-' * 94)
    print(f"{'eff':>5} {'TD3 mean':>12} {'SAC mean':>12}")
    for v in values:
        ta = [td3[(v, s)]['speed'] for s in seeds if (v, s) in td3]
        sa = [sac[(v, s)]['speed'] for s in seeds if (v, s) in sac]
        print(f"{v:>5.1f} {np.mean(ta):>12.1f} {np.mean(sa):>12.1f}")

    for name, rows in (('TD3 (original)', td3), ('SAC (this run)', sac)):
        r_o, p_o, r_w, per_seed = within_seed_stats(rows)
        print('-' * 94)
        print(f'{name}:')
        print(f'  Pearson r (all 27 runs)            = {r_o:+.4f}  (p = {p_o:.4f})')
        print(f'  within-seed (de-meaned) Pearson r  = {r_w:+.4f}')
        for s, (r_s, p_s) in per_seed:
            print(f'    seed {s:>4}: r = {r_s:+.4f} (p = {p_s:.4f})')

    print('-' * 94)
    for name, rows in (('TD3 (original)', td3_runs), ('SAC (this run)', sac_runs)):
        pl = plateaus(rows)
        print(f'{name} exact-repetition plateaus (same speed/safety/comfort triple):')
        if not pl:
            print('  NONE')
        for s, trip, vals in pl:
            print(f'  seed {s}: triple {trip} at efficiencies {vals}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
