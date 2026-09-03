"""Efficiency-style slider sweeps for SAFETY and COMFORT with real SAC.

Runs on the exact original failed-sweep test program (the ~50-iteration
loop(50) multi-sensor program). Same shape as the efficiency sweep:
9 slider values (0.1-0.9) x 3 seeds (42, 123, 999), 5000 steps/run,
50 eval episodes, safety/efficiency/comfort held at 0.5 except the one
being swept.

SWEEP A (--key safety):  safety slider; reward already uses the third
  safety design (speed_weight*(1-safety)*distance_covered). Reported vs the
  original TD3 baseline in tests/results/slider_sweep_safety_scaling.json.
SWEEP B (--key comfort): comfort slider; no prior TD3 baseline exists.

Usage:
  python tests/sac_slider_sweep.py sweep --key safety
  python tests/sac_slider_sweep.py sweep --key comfort
  python tests/sac_slider_sweep.py report --key safety
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.sac_verify import (RESULTS, OUT_DIR, run_training,
                              neutral_priorities, build_sweep_ir)  # noqa: E402

VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 123, 999]


def _run_path(key, value, seed):
    return OUT_DIR / ('%s_%.1f_seed_%s.json' % (key, value, seed))


def _load_runs(key):
    rows = []
    for v in VALUES:
        for s in SEEDS:
            p = _run_path(key, v, s)
            if p.exists():
                r = json.loads(p.read_text())
                r['slider'] = v
                rows.append(r)
    return rows


def cmd_sweep(args) -> None:
    key = args.key
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ir = build_sweep_ir()
    for value in args.values:
        for seed in args.seeds:
            out = _run_path(key, value, seed)
            if out.exists() and not args.force:
                print('%s=%.1f seed=%s: cached' % (key, value, seed))
                continue
            prio = neutral_priorities(**{key: value})
            start = time.time()
            res = run_training(ir, prio, seed=seed, algorithm='SAC',
                               trial_level=1)
            res['elapsed_s'] = round(time.time() - start, 2)
            out.write_text(json.dumps(res, indent=2))
            m = res['metrics']
            print('%s=%.1f seed=%s: %ss speed=%8.1f safety=%6.1f comfort=%6.1f '
                  'final_logA=%5.3f'
                  % (key, value, seed, res['elapsed_s'], m['speed'],
                     m['safety'], m['convenience'], res['final_logAlpha']))


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float('nan'), float('nan')
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _de_meaned(x, y, groups):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yy = y.astype(float).copy()
    for g in set(groups):
        idx = [i for i, v in enumerate(groups) if v == g]
        yy[idx] -= float(np.mean(y[idx]))
    return x, yy


def _analyse(key, runs):
    """Print summary + correlations + plateau check. Return a stats dict."""
    rows = sorted(runs, key=lambda r: (r['slider'], r['seed']))
    print('=' * 90)
    print('%s slider sweep (real SAC) - summary table (%d x %d runs)'
          % (key.capitalize(), len(VALUES), len(SEEDS)))
    print('=' * 90)
    print('%8s %6s %10s %8s %9s' % ('slider', 'seed', 'speed', 'safety%',
                                    'comfort%'))
    for r in rows:
        m = r['metrics']
        print('%7.1f %6d %10.1f %8.1f %9.1f'
              % (r['slider'], r['seed'], m['speed'], m['safety'],
                 m['convenience']))
    print('-' * 90)
    for v in VALUES:
        sel = [r for r in rows if abs(r['slider'] - v) < 1e-9]
        if not sel:
            continue
        sp = [r['metrics']['speed'] for r in sel]
        print('slider=%4.1f  mean speed %8.1f  (n=%d)' % (v, statistics.mean(sp),
                                                          len(sel)))
    print('-' * 90)
    xs = [r['slider'] for r in rows]
    groups = [r['seed'] for r in rows]
    stats_dict = {}
    for col in METRICS[key]:
        ys = [r['metrics'][col] for r in rows]
        r_all, p_all = _pearson(xs, ys)
        xd, yd = _de_meaned(xs, ys, groups)
        r_with, _ = _pearson(xd, yd)
        stats_dict[col] = {'r': r_all, 'p': p_all, 'r_within': r_with}
        print('metric %-11s: Pearson r = %+.4f (p = %.4f)   '
              'within-seed r = %+.4f'
              % (col, r_all, p_all, r_with))
        per_seed = {}
        for s in set(groups):
            idx = [i for i, g in enumerate(groups) if g == s]
            rs, ps = _pearson([xs[i] for i in idx], [ys[i] for i in idx])
            per_seed[s] = (rs, ps)
            print('    seed %3d: r = %+.4f (p = %.4f)' % (s, rs, ps))
        stats_dict[col]['per_seed'] = per_seed
    # exact-repetition plateau check (identical speed/safety/comfort triple)
    plateaus = []
    for s in set(groups):
        by = {}
        for r in rows:
            if r['seed'] != s:
                continue
            m = r['metrics']
            t = (round(m['speed'], 1), m['safety'], m['convenience'])
            by.setdefault(t, []).append(r['slider'])
        for t, vs in by.items():
            if len(vs) >= 2:
                plateaus.append((s, t, sorted(vs)))
    print('exact-repetition plateaus (>=2 sliders, identical triple): %d'
          % len(plateaus))
    for s, t, vs in sorted(plateaus, key=lambda z: -len(z[2])):
        print('  seed %d: triple %s at sliders %s' % (s, t, vs))
    stats_dict['plateaus'] = plateaus
    return stats_dict

# metric columns to correlate against the slider, per sweep key
METRICS = {
    'safety': ['safety', 'speed'],
    'comfort': ['convenience', 'speed'],
}

TD3_BASELINE = ROOT / 'tests' / 'results' / 'slider_sweep_safety_scaling.json'


def _td3_baseline_stats():
    """Stats for the original TD3 safety sweep (third reward design)."""
    data = json.loads(TD3_BASELINE.read_text())
    runs = data['runs']
    xs = [r['safety_slider'] for r in runs]
    groups = [r['seed'] for r in runs]
    out = {}
    for col in ('safety', 'speed'):
        ys = [r[col] for r in runs]
        r_all, p_all = _pearson(xs, ys)
        xd, yd = _de_meaned(xs, ys, groups)
        r_with, _ = _pearson(xd, yd)
        per_seed = {}
        for s in set(groups):
            idx = [i for i, g in enumerate(groups) if g == s]
            rs, ps = _pearson([xs[i] for i in idx], [ys[i] for i in idx])
            per_seed[s] = (rs, ps)
        out[col] = {'r': r_all, 'p': p_all, 'r_within': r_with,
                    'per_seed': per_seed}
    plateaus = []
    for s in set(groups):
        by = {}
        for r in runs:
            if r['seed'] != s:
                continue
            t = (round(r['speed'], 1), r['safety'], r['convenience'])
            by.setdefault(t, []).append(r['safety_slider'])
        for t, vs in by.items():
            if len(vs) >= 2:
                plateaus.append((s, t, sorted(vs)))
    out['plateaus'] = plateaus
    return out


def cmd_report(args) -> None:
    key = args.key
    runs = _load_runs(key)
    if len(runs) < 27:
        print('only %d/27 runs cached - rerun the sweep first' % len(runs))
        return
    stats_dict = _analyse(key, runs)
    agg = {'metadata': {
        'algorithm': 'SAC',
        'slider': key,
        'fixed': {'curiosity': 0.5},
        'values': VALUES,
        'steps_per_run': 5000,
        'eval_episodes': 50,
        'seeds': SEEDS,
        'program': 'ORIGINAL failed-TD3-sweep multi-sensor program (loop(50))',
    }, 'runs': [{
        key + '_slider': r['slider'], 'seed': r['seed'],
        'speed': r['metrics']['speed'], 'safety': r['metrics']['safety'],
        'convenience': r['metrics']['convenience'],
        'elapsed_s': r.get('elapsed_s'),
        'final_logAlpha': r.get('final_logAlpha')} for r in runs]}
    out = RESULTS / ('sac_%s_sweep.json' % key)
    out.write_text(json.dumps(agg, indent=2))
    print('saved %s' % out)

    if key == 'safety':
        print()
        print('---- TD3 baseline (original third-design safety sweep) ----')
        base = _td3_baseline_stats()
        for col in METRICS[key]:
            print('metric %-11s: Pearson r = %+.4f (p = %.4f)   '
                  'within-seed r = %+.4f'
                  % (col, base[col]['r'], base[col]['p'],
                     base[col]['r_within']))
            for s, (rs, ps) in sorted(base[col]['per_seed'].items()):
                print('    seed %3d: r = %+.4f (p = %.4f)' % (s, rs, ps))
        print('TD3 exact-repetition plateaus (>=2 sliders): %d'
              % len(base['plateaus']))
        for s, t, vs in sorted(base['plateaus'], key=lambda z: -len(z[2])):
            print('  seed %d: triple %s at sliders %s' % (s, t, vs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    p_sw = sub.add_parser('sweep')
    p_sw.add_argument('--key', required=True, choices=['safety', 'comfort'])
    p_sw.add_argument('--values', nargs='*', type=float, default=VALUES)
    p_sw.add_argument('--seeds', nargs='*', type=int, default=SEEDS)
    p_sw.add_argument('--force', action='store_true')
    p_sw.set_defaults(func=cmd_sweep)
    p_rep = sub.add_parser('report')
    p_rep.add_argument('--key', required=True, choices=['safety', 'comfort'])
    p_rep.set_defaults(func=cmd_report)
    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
