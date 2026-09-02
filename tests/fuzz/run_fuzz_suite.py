import argparse
import sys
import json
import os
import traceback
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from tests.fuzz.program_generator import generate_corpus
from parser import parse
from compiler import compile
from equivalence import check_equivalence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True, help='Number of programs to generate')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--max-depth', type=int, default=3, help='Max nesting depth')
    args = parser.parse_args()

    print(f"Generating {args.n} programs with seed {args.seed} and max depth {args.max_depth}...")
    corpus = generate_corpus(args.n, args.seed, args.max_depth)
    
    strict_pass_count = 0
    capped_equivalent_count = 0
    fail_count = 0
    failures_by_category = {}
    
    global_sensors_exercised = set()
    global_comparators_exercised = set()
    global_loop_forms = set()
    global_branch_forms = set()
    max_depth_reached = 0
    
    for i, (src, cov) in enumerate(corpus):
        global_sensors_exercised.update(cov['sensors_exercised'])
        global_comparators_exercised.update(cov['comparators_exercised'])
        global_loop_forms.update(cov['loop_forms_used'])
        global_branch_forms.update(cov['branch_forms_used'])
        if cov['max_depth_reached'] > max_depth_reached:
            max_depth_reached = cov['max_depth_reached']
            
        try:
            ast = parse(src)
            priorities = {'curiosity': 0.3, 'safety': 0.5, 'comfort': 0.5, 'efficiency': 0.5}
            ir = compile(ast, priorities)
            result = check_equivalence(ast, ir, max_steps=1000)
            
            if result['passed']:
                if result.get('capped_equivalent', False):
                    capped_equivalent_count += 1
                else:
                    strict_pass_count += 1
            else:
                fail_count += 1
                category = infer_category(cov)
                if category not in failures_by_category:
                    failures_by_category[category] = []
                failures_by_category[category].append({
                    'index': i,
                    'message': result['message'],
                    'src': src,
                    'features': cov
                })
        except Exception as e:
            fail_count += 1
            category = "Parse/Compile Error: " + type(e).__name__
            if category not in failures_by_category:
                failures_by_category[category] = []
            failures_by_category[category].append({
                'index': i,
                'message': str(e),
                'src': src,
                'features': cov,
                'traceback': traceback.format_exc()
            })

    total = args.n
    inclusive_pass_count = strict_pass_count + capped_equivalent_count
    strict_pass_rate = (strict_pass_count / total) * 100 if total > 0 else 0
    inclusive_pass_rate = (inclusive_pass_count / total) * 100 if total > 0 else 0
    real_fail_rate = (fail_count / total) * 100 if total > 0 else 0
    
    report = {
        'total_programs': total,
        'strict_pass_count': strict_pass_count,
        'capped_equivalent_count': capped_equivalent_count,
        'inclusive_pass_count': inclusive_pass_count,
        'fail_count': fail_count,
        'strict_pass_rate_pct': strict_pass_rate,
        'inclusive_pass_rate_pct': inclusive_pass_rate,
        'real_fail_rate_pct': real_fail_rate,
        'grammar_coverage': {
            'sensors_exercised_pct': len(global_sensors_exercised) / 7.0 * 100,
            'comparators_exercised_pct': len(global_comparators_exercised) / 6.0 * 100,
            'loop_forms_used': list(global_loop_forms),
            'branch_forms_used': list(global_branch_forms),
            'max_nesting_depth_reached': max_depth_reached
        },
        'failures_by_category': failures_by_category
    }

    # Print to stdout
    print("\n--- Fuzzing Summary Report ---")
    print(f"Total Programs:            {total}")
    print(f"Strict Passes (no cap):    {strict_pass_count} ({strict_pass_rate:.2f}%)")
    print(f"Capped-Equivalent Passes:  {capped_equivalent_count} ({(capped_equivalent_count / total * 100):.2f}%)")
    print(f"Inclusive Passes:          {inclusive_pass_count} ({inclusive_pass_rate:.2f}%)")
    print(f"Real Failures:             {fail_count} ({real_fail_rate:.2f}%)")
    print(f"Sensors exercised:         {len(global_sensors_exercised)}/7")
    print(f"Comparators exercised:     {len(global_comparators_exercised)}/6")
    print(f"Loop forms used:           {list(global_loop_forms)}")
    print(f"Branch forms used:         {list(global_branch_forms)}")
    print(f"Max depth reached:         {max_depth_reached}")
    
    if fail_count > 0:
        print("\nReal Failures by Category:")
        for cat, fails in failures_by_category.items():
            print(f"  - {cat}: {len(fails)} failures")
            
    # Write to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"tests/fuzz/results/fuzz_report_{args.seed}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport written to {filename}")

    sys.exit(0 if fail_count == 0 else 1)

def infer_category(features):
    if 'observe' in features['branch_forms_used'] and features['max_depth_reached'] >= 2:
        return "nested observe"
    if 'if' in features['branch_forms_used'] and features['max_depth_reached'] >= 2:
        return "nested if"
    if 'for' in features['loop_forms_used'] or 'until' in features['loop_forms_used']:
        return "loop logic error"
    return "straight-line execution error"

if __name__ == "__main__":
    main()
