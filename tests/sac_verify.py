"""SAC verification / sweep harness for Kobe live-training backend.

Gate-5 verification (task step 5): train real SAC on the ~50-step
multi-sensor test program via the exact production IPC bridge
(ipc/train_ipc.py spawned as a subprocess, as the Electron IDE does)
across 3 seeds, and report the reward curve over training steps, the
log-alpha trajectory (must move meaningfully from its init of 0), and the
final eval metrics.

NOTE / ASSUMPTION: the exact multi-sensor IR used by the earlier failed TD3
sweep is not stored in the repo (only its result JSONs under tests/results/
are). This harness therefore pins the multi-sensor test program in one place
so Gate 5 and the step-7 efficiency sweep share an identical reproducible
program: a 7-sensor program whose episodes run ~45-50 dynamic steps.

Usage:
  python tests/sac_verify.py run --seed 42
  python tests/sac_verify.py gate5
  python tests/sac_verify.py report
  python tests/sac_verify.py sweep --values 0.1 ... 0.9
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'tests' / 'results'
OUT_DIR = RESULTS / 'sac_verify'
sys.path.insert(0, str(ROOT))

GATE_SEEDS = [42, 123, 999]
SWEEP_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SWEEP_SEEDS = [42, 123, 999]

# -- The ~50-step multi-sensor test program (7 sensors) -------------------
MULTI_SENSOR_SOURCE = """
hardware {
  target: EV3
  motors: [A, B]
  sensors: [dist@1, colour@2, touch@3, IR@4, UV@5, gyro@6, sound@7]
}

policy {
  curiosity = 0.5;
  safety = 0.5;
  comfort = 0.5;
  efficiency = 0.5;
}

loop (2) {
  observe(dist, colour, touch, IR, UV, gyro, sound) {
    dist < 30 cm then { stop; }
    colour is red then { run forward; }
    touch pressed then { stop; }
    IR detected then { run forward; }
    UV index >= 5 then { walk forward slowly; }
    gyro tilt > 20 deg then { run forward; }
    sound >= 50 db then { walk forward; }
    else { run forward; }
  }
}
stop;
"""


# -- The ORIGINAL multi-sensor test program used by the FAILED TD3 sweeps -----
# (provided by the user for step 7; the trailing `else` belongs to the
# `gyro tilt > 45 deg` branch — whitespace is irrelevant to the tokenizer, so
# this text compiles to the exact IR the original sweep ran).
ORIGINAL_SWEEP_SOURCE = """
hardware {
  sensors: [dist@1, colour@2, touch@3, gyro@4]
}
policy {
  curiosity = 0.6;
  safety = 0.5;
  comfort = 0.5;
  efficiency = 0.8;
}
loop (50) {
  observe (dist, colour, touch, gyro) {
    touch pressed then {
      stop;
      wait 1 sec;
      turn left;
    }
    dist < 40 cm then {
      walk forward slowly;
    }
    colour is red then {
      turn right;
      walk forward;
    }
    gyro tilt > 45 deg then {
      walk forward slowly;
    }
    else {
      run forward quickly;
    }
  }
}
stop;
"""


def build_sweep_ir() -> list[dict]:
    """Compile the ORIGINAL failed-sweep program (step 7)."""
    from lexer import tokenize
    from parser import Parser
    from compiler import compile
    from priorities import DEFAULTS
    ast = Parser(tokenize(ORIGINAL_SWEEP_SOURCE)).parse()
    return compile(ast, DEFAULTS)


def build_multi_sensor_ir() -> list[dict]:
    """Compile the pinned program through the real lexer/parser/compiler."""
    from lexer import tokenize
    from parser import Parser
    from compiler import compile
    from priorities import DEFAULTS
    ast = Parser(tokenize(MULTI_SENSOR_SOURCE)).parse()
    return compile(ast, DEFAULTS)


def neutral_priorities(**overrides) -> dict:
    p = {'curiosity': 0.5, 'safety': 0.5, 'comfort': 0.5, 'efficiency': 0.5}
    p.update(overrides)
    return p


def run_training(ir, priorities, seed, algorithm='SAC', trial_level=1,
                 python=None, timeout=1800) -> dict:
    """Spawn ipc/train_ipc.py exactly like the IDE does; collect output."""
    req = {
        'ir': ir,
        'priorities': priorities,
        'algorithm': algorithm,
        'trialLevel': trial_level,
        'seed': int(seed),
    }
    proc = subprocess.run(
        [python or sys.executable, str(ROOT / 'ipc' / 'train_ipc.py')],
        input=json.dumps(req),
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            'train_ipc failed (seed=%s, alg=%s): %s'
            % (seed, algorithm, proc.stderr[-3000:]))
    steps, rewards, log_alphas, alphas = [], [], [], []
    done = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get('type') == 'progress':
            steps.append(msg['step'])
            rewards.append(msg['reward'])
            if 'logAlpha' in msg:
                log_alphas.append(msg['logAlpha'])
                alphas.append(msg['alpha'])
        elif msg.get('type') == 'done':
            done = msg
        elif msg.get('type') == 'error':
            raise RuntimeError('train_ipc error: %s' % msg.get('error'))
    if done is None:
        raise RuntimeError('no done message (seed=%s); stderr=%s'
                           % (seed, proc.stderr[-1500:]))
    return {
        'seed': int(seed),
        'steps': steps,
        'rewards': rewards,
        'logAlpha_traj': log_alphas,
        'alpha_traj': alphas,
        'metrics': done.get('metrics'),
        'final_logAlpha': done.get('logAlpha'),
        'final_alpha': done.get('alpha'),
        'targetEntropy': done.get('targetEntropy'),
        'algorithm': done.get('algorithm', algorithm),
        'trainingSteps': done.get('trainingSteps'),
    }


def _run_path(seed, tag='gate5') -> Path:
    return OUT_DIR / ('%s_seed_%s.json' % (tag, seed))


def cmd_run(args) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ir = build_multi_sensor_ir()
    start = time.time()
    res = run_training(ir, neutral_priorities(), seed=args.seed,
                       algorithm=args.algorithm, trial_level=args.trial_level)
    res['elapsed_s'] = round(time.time() - start, 2)
    out = _run_path(args.seed, tag=args.tag)
    out.write_text(json.dumps(res, indent=2))
    print('seed %s: %ss  metrics=%s  final logAlpha=%s alpha=%s'
          % (args.seed, res['elapsed_s'], res['metrics'],
             res['final_logAlpha'], res['final_alpha']))
    print('saved %s' % out)


def cmd_gate5(args) -> None:
    for seed in args.seeds or GATE_SEEDS:
        out = _run_path(seed, tag='gate5')
        if out.exists() and not args.force:
            print('seed %s: already present (%s) - use --force to rerun' % (seed, out))
            continue
        cmd_run(argparse.Namespace(seed=seed, algorithm='SAC', trial_level=1,
                                   tag='gate5'))


def _load(tag='gate5') -> list[dict]:
    if not OUT_DIR.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob('%s_seed_*.json' % tag))]


def cmd_report(args) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    runs = _load(args.tag)
    if not runs:
        print('no runs found; run `python tests/sac_verify.py gate5` first')
        return
    runs.sort(key=lambda r: r['seed'])
    print('=' * 78)
    print('GATE 5 - real SAC on the ~50-step multi-sensor test program '
          '(5000 steps, %d seeds)' % len(runs))
    print('=' * 78)
    print('%5s %9s %8s %9s %10s %8s %8s %18s'
          % ('seed', 'speed', 'safety', 'comfort', 'final logA', 'final a',
             'dlogA', 'a in [min,max]'))
    log_alphas = []
    for r in runs:
        traj = r.get('logAlpha_traj', [])
        delta = (traj[-1] - traj[0]) if len(traj) >= 2 else float('nan')
        mn, mx = (min(traj), max(traj)) if traj else (float('nan'), float('nan'))
        log_alphas.extend(traj)
        m = r.get('metrics', {})
        print('%5d %9.1f %8.1f %9.1f %10.4f %8.3f %8.4f [%8.3f,%7.3f]'
              % (r['seed'], m.get('speed', 0), m.get('safety', 0),
                 m.get('convenience', 0), r.get('final_logAlpha', float('nan')),
                 r.get('final_alpha', float('nan')), delta, mn, mx))
    print('-' * 78)
    if log_alphas:
        print('log-alpha across seeds: init = 0.0 (CleanRL autotune); '
              'observed range [%.4f, %.4f]' % (min(log_alphas), max(log_alphas)))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for r in runs:
        ax[0].plot(r['steps'], r['rewards'], marker='o', ms=3,
                   label='seed %s' % r['seed'])
        traj_steps = r['steps'][:len(r['logAlpha_traj'])]
        ax[1].plot(traj_steps, r['logAlpha_traj'], marker='o', ms=3,
                   label='seed %s' % r['seed'])
    ax[1].axhline(0.0, color='grey', lw=0.8, ls=':')
    ax[0].set_xlabel('training step')
    ax[0].set_ylabel('episode reward (per 500-step log window)')
    ax[0].set_title('SAC reward curve over training')
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('training step')
    ax[1].set_ylabel('log alpha (learned temperature)')
    ax[1].set_title('log-alpha trajectory over training')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    png = RESULTS / ('sac_%s_curves.png' % args.tag)
    fig.savefig(png, dpi=110)
    print('plots saved: %s' % png)

def cmd_sweep(args) -> None:
    """Step 7: 9 efficiency values x 3 seeds with real SAC (identical shape
    to the original failed TD3 sweep in slider_sweep_efficiency.json)."""
    import numpy as np
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ir = build_sweep_ir()  # ORIGINAL failed-TD3-sweep program (step 7)
    all_runs = {}
    for value in args.values:
        for seed in args.seeds:
            out = OUT_DIR / ('sweep_eff_%.1f_seed_%s.json' % (value, seed))
            if out.exists() and not args.force:
                all_runs[(value, seed)] = json.loads(out.read_text())
                print('eff=%.1f seed=%s: cached' % (value, seed))
                continue
            prio = neutral_priorities(efficiency=value)
            start = time.time()
            res = run_training(ir, prio, seed=seed, algorithm='SAC', trial_level=1)
            res['elapsed_s'] = round(time.time() - start, 2)
            res['efficiency_slider'] = value
            out.write_text(json.dumps(res, indent=2))
            all_runs[(value, seed)] = res
            print('eff=%.1f seed=%s: %ss speed=%.1f final_logA=%.4f'
                  % (value, seed, res['elapsed_s'], res['metrics']['speed'],
                     res['final_logAlpha']))
    rows = []
    for value in args.values:
        for seed in args.seeds:
            r = all_runs[(value, seed)]
            rows.append((value, seed, r['metrics']['speed'], r['metrics']['safety'],
                         r['metrics']['convenience']))
    xs = np.array([r[0] for r in rows])
    ys = np.array([r[2] for r in rows])
    print('=' * 78)
    print('STEP-7 EFFICIENCY SWEEP (real SAC) - summary table (9 x 3 = 27 runs)')
    print('=' * 78)
    print('%5s %6s %10s %8s %9s' % ('eff', 'seed', 'speed', 'safety', 'comfort'))
    for (v, s, sp, sa, co) in rows:
        print('%5.1f %6d %10.1f %8.1f %9.1f' % (v, s, sp, sa, co))
    print('-' * 78)
    print('%5s %12s %8s' % ('eff', 'mean speed', 'sd'))
    for v in args.values:
        vv = [r[2] for r in rows if r[0] == v]
        print('%5.1f %12.1f %8.1f' % (v, statistics.mean(vv), statistics.pstdev(vv)))
    print('-' * 78)
    r_pear, p_val = _pearson(xs, ys)
    print('Pearson r (all 27 runs) = %+.4f  (p = %.4f)' % (r_pear, p_val))
    demean = np.array(ys, dtype=float).copy()
    for s in args.seeds:
        mask = [i for i, r in enumerate(rows) if r[1] == s]
        mean_s = float(np.mean([ys[i] for i in mask]))
        for i in mask:
            demean[i] -= mean_s
    r_within = float(np.corrcoef(xs, demean)[0, 1])
    print('within-seed (de-meaned) Pearson r = %+.4f' % r_within)
    print('per-seed local-optimum check (identical eval outputs across slider values):')
    for s in args.seeds:
        speeds = [(r[0], r[2]) for r in rows if r[1] == s]
        plateaus = {}
        for (v, sp) in speeds:
            plateaus.setdefault(sp, []).append(v)
        hits = {sp: vs for sp, vs in plateaus.items() if len(vs) >= 3}
        if hits:
            for sp, vs in hits.items():
                print('  seed %s: FLAT at speed %.1f across efficiencies %s'
                      % (s, sp, vs))
        else:
            print('  seed %s: no >=3-value plateau detected (speed varies)' % s)
    out = RESULTS / 'sac_efficiency_sweep.json'
    out.write_text(json.dumps({'metadata': {
        'algorithm': 'SAC',
        'curiosity': 0.5, 'safety': 0.5, 'comfort': 0.5,
        'efficiency_values': args.values,
        'steps_per_run': 5000,
        'eval_episodes': 50,
        'seeds': args.seeds,
        'program': 'ORIGINAL failed-TD3-sweep multi-sensor program (user-supplied, loop(50))'},
        'runs': [{'efficiency_slider': r[0], 'seed': r[1],
                  'speed': round(r[2], 2), 'safety': r[3],
                  'convenience': r[4],
                  'elapsed_s': all_runs[(r[0], r[1])].get('elapsed_s'),
                  'final_logAlpha': all_runs[(r[0], r[1])].get('final_logAlpha')}
                 for r in rows]}, indent=2))
    print('saved %s' % out)


def _pearson(x, y):
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float('nan'), float('nan')
    from scipy import stats
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)

    p_run = sub.add_parser('run')
    p_run.add_argument('--seed', type=int, required=True)
    p_run.add_argument('--algorithm', default='SAC')
    p_run.add_argument('--trial-level', type=int, default=1)
    p_run.add_argument('--tag', default='gate5')
    p_run.set_defaults(func=cmd_run)

    p_g5 = sub.add_parser('gate5')
    p_g5.add_argument('--seeds', nargs='*', type=int)
    p_g5.add_argument('--force', action='store_true')
    p_g5.set_defaults(func=cmd_gate5)

    p_rep = sub.add_parser('report')
    p_rep.add_argument('--tag', default='gate5')
    p_rep.set_defaults(func=cmd_report)

    p_sw = sub.add_parser('sweep')
    p_sw.add_argument('--values', nargs='*', type=float, default=SWEEP_VALUES)
    p_sw.add_argument('--seeds', nargs='*', type=int, default=SWEEP_SEEDS)
    p_sw.add_argument('--force', action='store_true')
    p_sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
