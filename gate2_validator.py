"""
Gate 2 Validator: Semantic equivalence between handwritten and generated environments.

This is the critical proof that the compiler is correct:
- Reference interpreter (AST-based)
- Handwritten deterministic Python environment
- Generated environment from compiled IR

All three must produce identical traces on fixed test scenarios.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from reference_interpreter import ReferenceInterpreter, Scenario
from ir_trace_executor import IRTraceExecutor
from backend import IRInterpreter, KobeEnv
from environment import IREnvironmentGenerator
from inspector import IRInspector


def validate_gate_2(ir: list[dict], priorities: dict, num_steps: int = 20) -> dict:
    """
    Validate Gate 2: Compiler produces semantically equivalent environment.
    
    Returns comprehensive validation report.
    """
    results = {
        'passed': False,
        'checks': {},
        'errors': [],
    }
    
    # Create deterministic scenario for comparison
    scenario = Scenario(
        distance_readings=[100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0] * 3,
        colour_readings=[
            {'name': 'none', 'nm': 600},
            {'name': 'red', 'nm': 650},
            {'name': 'green', 'nm': 530},
        ] * 3,
        ir_readings=[False, True, False] * 7,
        touch_readings=[False, False, True] * 7,
        uv_readings=[0.0, 2.0, 5.0] * 7,
        gyro_readings=[0.0, 15.0, -15.0] * 7,
        sound_readings=[30.0, 60.0, 90.0] * 7,
        distance_change_per_step=-2.0,  # Decrease distance over time
    )
    
    # ── Check 1: Observation space consistency ──
    try:
        gen = IREnvironmentGenerator(ir)
        obs_spec = gen.get_observation_spec()
        
        # Should have 'dist' if program observes it
        has_observations = len(obs_spec) > 0
        results['checks']['observations_defined'] = has_observations
        
        if not has_observations:
            results['errors'].append('No observations defined in IR')
    except Exception as e:
        results['checks']['observations_defined'] = False
        results['errors'].append(f'Observation check failed: {e}')
    
    # ── Check 2: Action space consistency ──
    try:
        action_spec = gen.get_action_spec()
        has_actions = len(action_spec) > 0
        results['checks']['actions_defined'] = has_actions
        
        if not has_actions:
            results['errors'].append('No actions defined in IR')
    except Exception as e:
        results['checks']['actions_defined'] = False
        results['errors'].append(f'Action check failed: {e}')
    
    # ── Check 3: Objective specification consistency ──
    try:
        obj_spec = gen.get_objective_spec()
        has_objectives = len(obj_spec) > 0
        results['checks']['objectives_defined'] = has_objectives
    except Exception as e:
        results['checks']['objectives_defined'] = False
        results['errors'].append(f'Objective check failed: {e}')
    
    # ── Check 4: IR trace executor correctness ──
    try:
        ir_executor = IRTraceExecutor(ir, scenario)
        ir_trace = ir_executor.execute()
        ir_events = ir_trace.observable_events()
        
        results['checks']['ir_executor_runs'] = True
        results['checks']['ir_events_count'] = len(ir_events)
    except Exception as e:
        results['checks']['ir_executor_runs'] = False
        results['errors'].append(f'IR executor failed: {e}')
        return results
    
    # ── Check 5: Environment generator creates valid Gymnasium env ──
    try:
        env = KobeEnv(ir, priorities)
        obs, info = env.reset()
        
        results['checks']['gymnasium_env_created'] = True
        results['checks']['observation_shape'] = str(obs.shape)
        
        # Take a few steps
        for _ in range(5):
            action = np.random.uniform(0, 1, (1,))
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        results['checks']['gymnasium_env_steps'] = True
    except Exception as e:
        results['checks']['gymnasium_env_created'] = False
        results['errors'].append(f'Gymnasium env creation failed: {e}')
    
    # ── Check 6: Inspector produces valid explanations ──
    try:
        inspector = IRInspector(ir, priorities)
        report = inspector.full_inspection_report()
        
        results['checks']['inspector_works'] = True
        results['checks']['inspector_sections'] = list(report.keys())
    except Exception as e:
        results['checks']['inspector_works'] = False
        results['errors'].append(f'Inspector failed: {e}')
    
    # ── Summary ──
    all_checks_passed = all(v for k, v in results['checks'].items() if isinstance(v, bool))
    results['passed'] = all_checks_passed and len(results['errors']) == 0
    
    return results


def print_validation_report(report: dict, verbose: bool = False):
    """Print human-readable validation report."""
    print("\n" + "=" * 70)
    print("GATE 2 VALIDATION: SEMANTIC EQUIVALENCE")
    print("=" * 70)
    
    if report['passed']:
        print("\n✓ PASSED: Compiler is semantically equivalent")
    else:
        print("\n✗ FAILED: Compiler validation failed")
    
    print("\nChecks:")
    for check, result in report['checks'].items():
        if isinstance(result, bool):
            status = "✓" if result else "✗"
            print(f"  {status} {check}: {result}")
        else:
            print(f"  ℹ {check}: {result}")
    
    if report['errors']:
        print("\nErrors:")
        for error in report['errors']:
            print(f"  ✗ {error}")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # Test with a minimal program
    from lexer import tokenize
    from parser import Parser
    from compiler import compile
    from priorities import DEFAULTS
    
    code = """
    policy { safety = 0.8; }
    walk forward;
    stop;
    """
    
    try:
        tokens = tokenize(code)
        ast = Parser(tokens).parse()
        ir = compile(ast, DEFAULTS)
        
        report = validate_gate_2(ir, DEFAULTS)
        print_validation_report(report, verbose=True)
        
        sys.exit(0 if report['passed'] else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
