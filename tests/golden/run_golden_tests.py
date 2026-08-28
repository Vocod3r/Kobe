"""
Golden program test harness: run semantic equivalence tests.

For each golden program:
  1. Parse it
  2. Create a deterministic scenario
  3. Run reference interpreter (AST-based) and get trace
  4. Compile to IR
  5. Run IR trace executor against same scenario
  6. Compare observable events (action/observe/branch/break/halt)
  7. Report pass/fail

Pass = traces match exactly on all observable events.
Fail = traces diverge, indicating a compiler or executor bug.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lexer import tokenize
from parser import Parser
from compiler import compile
from semantic_analyzer import SemanticAnalyzer
from equivalence import default_scenario, check_equivalence
from tests.golden.golden_programs import GOLDEN_PROGRAMS


def run_golden_tests():
    """Run all golden program tests and report results."""
    results = {
        'passed': [],
        'failed': [],
        'errors': [],
    }
    
    for name, source_code in GOLDEN_PROGRAMS.items():
        try:
            passed, message = test_semantic_equivalence(name, source_code)
            if passed:
                results['passed'].append((name, message))
                print(f"PASS: {name}")
            else:
                results['failed'].append((name, message))
                print(f"FAIL: {name}: {message}")
        except Exception as e:
            results['errors'].append((name, str(e)))
            print(f"ERR : {name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(results['passed'])} passed, {len(results['failed'])} failed, {len(results['errors'])} errors")
    print("=" * 60)
    
    return results


def test_semantic_equivalence(name: str, source_code: str) -> tuple[bool, str]:
    """
    Test a single program for semantic equivalence.
    
    Returns (passed: bool, message: str)
    """
    # Parse
    try:
        tokens = tokenize(source_code)
        parser = Parser(tokens)
        ast = parser.parse()
    except Exception as e:
        return False, f"Parse error: {e}"
    
    # Semantic analysis
    analyzer = SemanticAnalyzer()
    diagnostics = analyzer.analyze(ast)
    
    # Check for errors (warnings/hints are OK)
    errors = [d for d in diagnostics if d.severity == 'error']
    if errors:
        return False, f"Semantic analysis errors: {errors[0].message}"
    
    # Compile
    try:
        from priorities import DEFAULTS
        priorities = DEFAULTS.copy()
        ir = compile(ast, priorities)
    except Exception as e:
        return False, f"Compile error: {e}"
    
    # Run the shared reference-interpreter vs IR-trace-executor equivalence check
    # (same scenario + comparator gate2_validator uses, so results can't drift).
    result = check_equivalence(ast, ir, scenario=default_scenario())
    if result['ref_event_count'] is None or result['ir_event_count'] is None:
        # One side crashed before producing any events.
        return False, result['message']
    return result['passed'], result['message']


if __name__ == '__main__':
    results = run_golden_tests()
    
    # Exit with status code
    exit_code = 0 if len(results['failed']) == 0 and len(results['errors']) == 0 else 1
    sys.exit(exit_code)