"""
Deep analysis of the loop-failure cases using the canonical equivalence.py implementation.

For each of the 200 fuzzed programs (seed=42, max-depth=3):
  - Uses canonical check_equivalence(ast, ir, scenario=scenario, max_steps=1000)
  - Buckets results into:
      Strict Pass (neither hit cap, traces identical)
      Capped-Equivalent (both hit cap, shared prefix identical)
      Real Failures (divergence or termination disagreement)
  - Separately analyses satisfiability of loop_until conditions against the
    default scenario values
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.fuzz.program_generator import generate_corpus
from parser import parse
from compiler import compile
from equivalence import default_scenario, check_equivalence
from reference_interpreter import ReferenceInterpreter, Scenario
from ir_trace_executor import IRTraceExecutor

MAX_STEPS = 1000


# ── Satisfiability analysis ───────────────────────────────────────────────────

from compiler import _to_cm

SCENARIO = default_scenario()

def _sensor_values():
    return {
        'dist':   SCENARIO.distance_readings,
        'colour': [c['name'] for c in SCENARIO.colour_readings],
        'IR':     SCENARIO.ir_readings,
        'touch':  SCENARIO.touch_readings,
        'UV':     SCENARIO.uv_readings,
        'gyro':   SCENARIO.gyro_readings,
        'sound':  SCENARIO.sound_readings,
    }

def _cmp(a, op, b):
    return {'<': a<b,'<=': a<=b,'>': a>b,'>=': a>=b,'==': a==b,'!=': a!=b}.get(op, False)

def _condition_satisfiable(cond: dict) -> bool:
    vals = _sensor_values()
    t = cond['type']

    if t == 'DistCondition':
        cm = _to_cm(cond['value'], cond['unit'])
        return any(_cmp(d, cond['comparator'], cm) for d in vals['dist'])

    if t == 'ColourCondition':
        colours = vals['colour']
        if isinstance(cond['colour'], str):
            results = [c == cond['colour'] for c in colours]
        else:
            results = [True] * len(colours)
        if cond['negate']:
            results = [not r for r in results]
        return any(results)

    if t == 'TouchCondition':
        return any(vals['touch'])

    if t == 'IRCondition':
        if cond['mode'] == 'detected':
            return any(vals['IR'])
        return any(_cmp(v, cond['comparator'], cond['value']) for v in vals['IR'])

    if t == 'UVCondition':
        if cond['mode'] == 'detected':
            return any(v > 0 for v in vals['UV'])
        return any(_cmp(v, cond['comparator'], cond['index']) for v in vals['UV'])

    if t == 'GyroCondition':
        return any(_cmp(v, cond['comparator'], cond['degrees']) for v in vals['gyro'])

    if t == 'SoundCondition':
        return any(_cmp(v, cond['comparator'], cond['db']) for v in vals['sound'])

    if t == 'And':
        return _condition_satisfiable(cond['left']) and _condition_satisfiable(cond['right'])

    if t == 'Or':
        return _condition_satisfiable(cond['left']) or _condition_satisfiable(cond['right'])

    if t == 'Not':
        return not _condition_satisfiable(cond['operand'])

    return False


def _collect_loop_until_conditions(body: list) -> list:
    conds = []
    for stmt in body:
        if stmt['type'] == 'LoopUntil':
            conds.append(stmt['condition'])
        for sub in _sub_bodies(stmt):
            conds.extend(_collect_loop_until_conditions(sub))
    return conds


def _sub_bodies(stmt: dict) -> list:
    bodies = []
    if 'body' in stmt:
        bodies.append(stmt['body'])
    if 'then' in stmt:
        bodies.append(stmt['then'])
    if 'else' in stmt and stmt['else']:
        bodies.append(stmt['else'])
    if stmt['type'] == 'Observe':
        for branch in stmt.get('branches', []):
            bodies.append(branch['then'])
            if branch.get('else'):
                bodies.append(branch['else'])
    return bodies


def main():
    print("Generating corpus (seed=42, n=200, max-depth=3)...")
    corpus = generate_corpus(200, 42, max_depth=3)
    priorities = {'curiosity': 0.3, 'safety': 0.5, 'comfort': 0.5, 'efficiency': 0.5}
    scenario = default_scenario()

    strict_passes = []
    capped_equivalent_passes = []
    real_failures = []

    for i, (src, cov) in enumerate(corpus):
        try:
            ast = parse(src)
            ir = compile(ast, priorities)
            res = check_equivalence(ast, ir, scenario=scenario, max_steps=MAX_STEPS)
        except Exception as e:
            real_failures.append({
                'index': i,
                'message': f"Exception: {e}",
                'src': src,
                'features': cov
            })
            continue

        if res['passed']:
            if res.get('capped_equivalent', False):
                capped_equivalent_passes.append({
                    'index': i,
                    'message': res['message'],
                    'src': src,
                    'features': cov
                })
            else:
                strict_passes.append({
                    'index': i,
                    'message': res['message'],
                    'src': src,
                    'features': cov
                })
        else:
            real_failures.append({
                'index': i,
                'message': res['message'],
                'src': src,
                'features': cov,
                'ref_hit_cap': res.get('ref_hit_cap'),
                'ir_hit_cap': res.get('ir_hit_cap'),
                'ref_event_count': res.get('ref_event_count'),
                'ir_event_count': res.get('ir_event_count'),
            })

    total = len(corpus)
    print(f"\n{'='*70}")
    print(f"CANONICAL FUZZ RESULTS BREAKDOWN (n={total})")
    print(f"{'='*70}")
    print(f"  Strict Passes (no cap reached):      {len(strict_passes)} ({(len(strict_passes)/total*100):.2f}%)")
    print(f"  Capped-Equivalent Passes (matched):  {len(capped_equivalent_passes)} ({(len(capped_equivalent_passes)/total*100):.2f}%)")
    print(f"  Inclusive Passes (Strict + Capped):  {len(strict_passes) + len(capped_equivalent_passes)} ({((len(strict_passes) + len(capped_equivalent_passes))/total*100):.2f}%)")
    print(f"  Real Failures:                       {len(real_failures)} ({(len(real_failures)/total*100):.2f}%)")

    # Satisfiability analysis
    total_until = 0
    satisfiable_until = 0
    unsatisfiable_until = 0
    programs_with_until = 0
    programs_all_unsat = 0

    for src, cov in corpus:
        if 'until' not in cov.get('loop_forms_used', []):
            continue
        try:
            ast = parse(src)
        except Exception:
            continue
        programs_with_until += 1
        conds = _collect_loop_until_conditions(ast['body'])
        total_until += len(conds)
        sat_conds = [_condition_satisfiable(c) for c in conds]
        satisfiable_until += sum(sat_conds)
        unsatisfiable_until += sum(not s for s in sat_conds)
        if conds and not any(sat_conds):
            programs_all_unsat += 1

    print(f"\n{'='*70}")
    print(f"LOOP_UNTIL SATISFIABILITY ANALYSIS")
    print(f"{'='*70}")
    print(f"  Programs with at least one loop_until:    {programs_with_until}")
    print(f"  Total loop_until conditions examined:     {total_until}")
    sat_pct = 100*satisfiable_until/total_until if total_until else 0
    unsat_pct = 100*unsatisfiable_until/total_until if total_until else 0
    print(f"  Satisfiable (exit reachable in scenario): {satisfiable_until} ({sat_pct:.1f}%)")
    print(f"  Unsatisfiable (loop can NEVER exit):      {unsatisfiable_until} ({unsat_pct:.1f}%)")
    print(f"  Programs where ALL loop_until are unsat:  {programs_all_unsat} / {programs_with_until}")


if __name__ == '__main__':
    main()
