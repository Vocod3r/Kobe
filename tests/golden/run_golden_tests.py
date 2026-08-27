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
from reference_interpreter import ReferenceInterpreter, Scenario
from ir_trace_executor import IRTraceExecutor
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
    
    # Create scenario (deterministic sensor readings)
    scenario = create_scenario_for_program(ast)
    
    # Execute with reference interpreter
    try:
        ref_interp = ReferenceInterpreter(ast, scenario)
        ref_trace = ref_interp.execute()
        ref_events = ref_trace.observable_events()
    except Exception as e:
        return False, f"Reference interpreter error: {e}"
    
    # Execute with IR trace executor
    try:
        ir_executor = IRTraceExecutor(ir, scenario)
        ir_trace = ir_executor.execute()
        ir_events = ir_trace.observable_events()
    except Exception as e:
        return False, f"IR executor error: {e}"
    
    # Compare traces
    return compare_traces(ref_events, ir_events, name)


def create_scenario_for_program(ast: dict) -> Scenario:
    """Create a deterministic scenario for a program based on its structure."""
    # Default scenario: steady state with distance far from thresholds
    scenario = Scenario(
        distance_readings=[100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 15.0],
        colour_readings=[
            {'name': 'red', 'nm': 650},
            {'name': 'green', 'nm': 530},
            {'name': 'blue', 'nm': 470},
            {'name': 'none', 'nm': 600},
        ],
        ir_readings=[False, False, True, False],
        touch_readings=[False, False, False, True],
        uv_readings=[0.0, 2.0, 5.0, 8.0],
        gyro_readings=[0.0, 15.0, -15.0, 30.0],
        sound_readings=[30.0, 50.0, 70.0, 90.0],
        distance_change_per_step=0.0,  # Fixed for determinism
    )
    
    return scenario


def compare_traces(ref_events, ir_events, program_name: str) -> tuple[bool, str]:
    """
    Compare two traces for equivalence.
    
    Observable events: action, observe, branch, break, halt.
    loop_iter is informational only (not compared).
    
    Returns (passed: bool, message: str)
    """
    if len(ref_events) != len(ir_events):
        return False, f"Event count mismatch: ref={len(ref_events)}, ir={len(ir_events)}"
    
    for i, (ref_evt, ir_evt) in enumerate(zip(ref_events, ir_events)):
        # Check event kind
        if ref_evt.kind != ir_evt.kind:
            return False, f"Event {i} kind mismatch: ref={ref_evt.kind}, ir={ir_evt.kind}"
        
        # For action/observe/branch events, compare key details
        if ref_evt.kind == 'action':
            if ref_evt.details.get('action') != ir_evt.details.get('action'):
                return False, f"Event {i} action mismatch: ref={ref_evt.details}, ir={ir_evt.details}"
        
        elif ref_evt.kind == 'observe':
            if ref_evt.details.get('sensors') != ir_evt.details.get('sensors'):
                return False, f"Event {i} observe sensors mismatch"
            # Readings should match (same scenario)
            ref_readings = ref_evt.details.get('readings', {})
            ir_readings = ir_evt.details.get('readings', {})
            for sensor in ref_readings:
                if ref_readings.get(sensor) != ir_readings.get(sensor):
                    return False, f"Event {i} reading mismatch for {sensor}: ref={ref_readings.get(sensor)}, ir={ir_readings.get(sensor)}"
        
        elif ref_evt.kind == 'branch':
            # Branch result (True/False) must match
            if ref_evt.details.get('result') != ir_evt.details.get('result'):
                return False, f"Event {i} branch result mismatch: ref={ref_evt.details.get('result')}, ir={ir_evt.details.get('result')}"
    
    return True, "All events match"


if __name__ == '__main__':
    results = run_golden_tests()
    
    # Exit with status code
    exit_code = 0 if len(results['failed']) == 0 and len(results['errors']) == 0 else 1
    sys.exit(exit_code)
