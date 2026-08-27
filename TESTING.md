# Kobe Testing & Validation Guide

Complete reference for testing the Kobe compiler and generated artifacts.

## Overview

Kobe's testing strategy proves correctness at multiple gates:

- **Gate 1 (Syntax & Parsing)**: Parser correctly tokenizes and builds AST
- **Gate 2 (Compiler Correctness)**: Generated IR is semantically equivalent to reference semantics
- **Gate 3 (Environment Generation)**: Compiled IR → valid Gymnasium environment
- **Gate 4 (Training Pipeline)**: Generated environment + trainer works end-to-end
- **Gate 5 (Study Readiness)**: System is ready for human user evaluation

## Built Components

### 1. Reference Interpreter (`reference_interpreter.py`)
- Walks the AST directly (tree recursion, not jumps)
- Takes deterministic sensor scenarios
- Produces traces of actions/observations/branches
- **Independent of the compiler** — catches compiler bugs structurally

### 2. IR Trace Executor (`ir_trace_executor.py`)
- Walks compiled IR (jump-threaded)
- Takes the same scenario as reference interpreter
- Produces the same trace structure
- Two different representations should agree on observable behavior

### 3. Handwritten Baseline (`handwritten_baseline.py`)
- Deterministic proportional controller
- No learning involved
- Scales aggressiveness with clearance (distance)
- Provides floor for comparison: if hand-written gets 95%, trained 96% is meaningless

### 4. Golden Program Tests (`tests/golden/run_golden_tests.py`)
- 10 golden programs covering all compiler features
- Compares reference interpreter vs IR executor traces
- Pass = semantic equivalence confirmed
- Fail = compiler or executor bug

### 5. Environment Generator (`environment.py`) [NEW]
- Compiles IR to Gymnasium-compatible environments
- Analyzes IR for sensor/action/objective specs
- Produces valid obs/action spaces
- Basis for environment-based testing and training

### 6. Trainer Generator (`trainer.py`) [NEW]
- Generates TorchRL training configurations from IR
- Selects algorithm (SAC default) and hyperparameters
- Creates executable training scripts
- Ensures training respects safety constraints and weighted objectives

### 7. Inspector (`inspector.py`) [NEW]
- Explains generated environment/training to users
- Produces human-readable explanations
- Critical for transparency/inspectability research question
- Generates full inspection reports and pretty-prints

### 8. Study Tasks (`study_tasks.py`) [NEW]
- Three matched tasks for human evaluation (T1, T2, T3)
- Transfer task for generalization testing
- Kobe and Python baseline solutions provided
- Evaluation rubric and success criteria

## Quick Start in VS Code

### Option 1: Run core tests in terminal

Open a terminal in VS Code (`Ctrl+~` or `View > Terminal`):

```powershell
# 1. Run golden program tests (Gate 2: Semantic equivalence)
python tests/golden/run_golden_tests.py

# 2. Validate Gate 2 (compiler correctness)
python gate2_validator.py

# 3. Compare baselines (random vs handwritten)
python compare_baselines.py

# 4. Compare baselines including trained policy (optional)
python compare_baselines.py --trained --actor-path checkpoints/sac_actor.pt

# 5. Inspect generated artifacts
python -c "from inspector import print_inspection_report; from compiler import compile; from lexer import tokenize; from parser import Parser; from priorities import DEFAULTS; code='walk forward; stop;'; tokens=tokenize(code); ast=Parser(tokens).parse(); ir=compile(ast, DEFAULTS); print_inspection_report(ir, DEFAULTS)"
```

### Option 2: Interactive Python testing

In VS Code Python REPL or Python file:

```python
# Test single golden program
from tests.golden.run_golden_tests import test_semantic_equivalence
from tests.golden.golden_programs import GOLDEN_PROGRAMS

passed, msg = test_semantic_equivalence('simple_walk', GOLDEN_PROGRAMS['simple_walk'])
print(f"Result: {passed}, {msg}")

# Generate and inspect environment
from compiler import compile
from environment import IREnvironmentGenerator
from lexer import tokenize
from parser import Parser
from priorities import DEFAULTS

code = "walk forward; stop;"
tokens = tokenize(code)
ast = Parser(tokens).parse()
ir = compile(ast, DEFAULTS)

gen = IREnvironmentGenerator(ir)
print(f"Observations: {gen.get_observation_spec()}")
print(f"Actions: {gen.get_action_spec()}")
print(f"Objectives: {gen.get_objective_spec()}")

# Inspect what Kobe generated
from inspector import IRInspector
inspector = IRInspector(ir, DEFAULTS)
report = inspector.full_inspection_report()
print(f"Generated: {report['ir_summary']}")

# Generate training config
from trainer import TrainerGenerator
trainer_gen = TrainerGenerator(ir, DEFAULTS)
spec = trainer_gen.get_training_spec()
print(f"Training: {spec['algorithm']} with {spec['hyperparameters']['num_training_steps']} steps")
```

### Option 3: Study task testing

```python
from study_tasks import get_task_description, STUDY_TASKS

# View all study tasks
print(get_task_description('T1_obstacle'))
print(get_task_description('T2_target'))
print(get_task_description('T3_tradeoff'))

# Get task source code
task = STUDY_TASKS['T1_obstacle']
print(task['kobe'])  # Reference solution
print(task['python_scaffold'])  # What baseline programmers get
```

## Test Structure

### Golden Programs (10 tests)
Each tests a specific compiler feature:

1. **simple_walk** — Basic action sequence
2. **observe** — Reading sensor values
3. **if_distance** — Simple if-then-else
4. **if_chain** — Multiple else-if branches
5. **loop_for** — Fixed-count loop
6. **loop_until** — Loop with condition
7. **loop_for_with_break** — Break statement
8. **and_or_not_combinators** — Boolean logic
9. **not_combinator** — Negation operator
10. **nested_control** — Complex nested flow

Each test:
- Runs through reference interpreter (AST-based)
- Compiles to IR
- Runs IR through trace executor
- Compares observable events (action/observe/branch/break/halt)
- **Pass** = traces match exactly
- **Fail** = compiler or executor bug

### Baseline Comparison

Compares three controllers on same task:

| Metric | Random | Handwritten | Trained |
|--------|--------|-------------|---------|
| Speed | ~700 cm | ~600 cm | ~370 cm (trades speed for safety) |
| Safety | ~99.8% | 100% | 100% |
| Comfort | ~23% | ~25% | ~38% (smoother) |

Expected behavior:
- Random: fastest but least safe
- Handwritten: good baseline, deterministic, no learning
- Trained: should outperform handwritten on safety/comfort

If handwritten already gets 99% safety, trained only gets 99.1%, that's not a win.

## What Each Test Validates

### Golden Tests (Gate 2)
✓ Parser produces correct AST  
✓ Compiler produces correct IR  
✓ Branch condition handling (including negation)  
✓ Loop structures (for, until, break)  
✓ Combined conditions (and/or/not)  
✓ Sensor observation timing  
✓ Control flow (if/else/else-if)  

### Baseline Tests (Gate 2 Extended)
✓ KobeEnv works correctly  
✓ Handwritten controller is deterministic  
✓ Evaluation harness is consistent  
✓ Trained policies integrate with same harness  

### Gate 2 Validator
✓ Observations defined in IR  
✓ Actions defined in IR  
✓ Objectives extracted from IR  
✓ IR trace executor runs successfully  
✓ Gymnasium environment can be created  
✓ Inspector can analyze IR  

### Environment Generator (Gate 3)
✓ IR sensor specs map to Gymnasium spaces  
✓ Action specs convert to valid action spaces  
✓ Observation space matches program intent  
✓ Reset and step functions work without error  

### Trainer Generator (Gate 4)
✓ Algorithm selection works (SAC default)  
✓ Hyperparameters generated based on priorities  
✓ Training script produces valid Python code  
✓ Evaluation protocol is consistent  

### Inspector (Transparency)
✓ Can explain all observations  
✓ Can explain all actions  
✓ Can explain all objectives  
✓ Can explain termination conditions  
✓ Can explain safety constraints  
✓ Can explain training approach  

### Study Tasks (Gate 5)
✓ T1 tests observation+action+safety  
✓ T2 tests observation+goal+objective  
✓ T3 tests multi-objective tradeoff  
✓ Transfer tests generalization  
✓ Matched difficulty/concepts between Kobe and Python  

## Debugging Failed Tests

If a golden test fails:

```python
# (In Python REPL or debugger)
from tests.golden.run_golden_tests import test_semantic_equivalence
from tests.golden.golden_programs import GOLDEN_PROGRAMS

# Get detailed info
passed, msg = test_semantic_equivalence('loop_for', GOLDEN_PROGRAMS['loop_for'])
print(msg)  # Will show exactly where traces diverged

# To inspect the traces:
from lexer import tokenize
from parser import Parser
from compiler import compile
from reference_interpreter import ReferenceInterpreter, Scenario
from ir_trace_executor import IRTraceExecutor

code = GOLDEN_PROGRAMS['loop_for']
tokens = tokenize(code)
ast = Parser(tokens).parse()
ir = compile(ast, {})

scenario = Scenario()
ref = ReferenceInterpreter(ast, scenario)
ref_trace = ref.execute()

executor = IRTraceExecutor(ir, scenario)
ir_trace = executor.execute()

# Compare
for i, (ref_evt, ir_evt) in enumerate(zip(ref_trace.observable_events(), ir_trace.observable_events())):
    if ref_evt.kind != ir_evt.kind:
        print(f"Event {i} mismatch: ref={ref_evt.kind}, ir={ir_evt.kind}")
        break
```

## Integration with Trained Policies

To include trained SAC/TD3/DroQ policies in baseline comparison:

```python
# After training (backend.py)
# The compare_baselines.py already supports this:
python compare_baselines.py --trained --actor-path checkpoints/my_sac_actor.pt

# This loads the trained actor and evaluates it through the same
# KobeEnv/metrics harness, making results directly comparable
```

## Running All Tests

Create a simple test runner script (or use this in REPL):

```python
import subprocess
import sys

print("Running golden program tests...")
result1 = subprocess.run([sys.executable, 'tests/golden/run_golden_tests.py'])

print("\nRunning baseline comparison...")
result2 = subprocess.run([sys.executable, 'compare_baselines.py'])

if result1.returncode == 0 and result2.returncode == 0:
    print("\n✓ All tests passed!")
else:
    print("\n✗ Some tests failed")
    sys.exit(1)
```

## Notes

- All tests are deterministic (same input = same output)
- Golden tests should always pass (they test compiler correctness)
- Baseline scores will vary slightly due to randomized policy evaluation
- To get consistent baseline numbers, use same seed: `env.reset(seed=42)`
- Reference interpreter and IR executor are independent implementations — agreement between them is strong evidence of correctness

## New Features (Aug 28-30 Phase)

### Environment Generation (`environment.py`)
Compiles IR to Gymnasium environments:
- `IREnvironmentGenerator`: Analyzes IR, creates spec
- `create_kobe_environment()`: Factory for final env
- Validates observation/action/objective spaces
- Ground truth: matches reference interpreter behavior on scenarios

### Training Generation (`trainer.py`)
Generates TorchRL configs from IR:
- `TrainerGenerator`: Extracts training requirements
- `generate_training_script()`: Outputs executable Python
- `generate_hyperparameters()`: Tuned based on priorities
- Algorithm selection: SAC (default), TD3, DroQ support

### Inspection & Transparency (`inspector.py`)
Explains generated artifacts to users:
- `IRInspector`: Full analysis of IR
- `print_inspection_report()`: Terminal pretty-print
- Explains observations, actions, objectives, safety, training
- Critical for research RQ4: does transparency help debugging?

### Study Tasks (`study_tasks.py`)
Three matched tasks for controlled study:
- **T1 (Obstacle)**: Observation + Action + Safety (10-15 min)
- **T2 (Target)**: Multiple observations + Goal + Objective (20-25 min)
- **T3 (Tradeoff)**: Multi-objective optimization (25-30 min)
- **Transfer**: Unseen task for generalization
- Both Kobe and Python solutions provided

### Gate 2 Validator (`gate2_validator.py`)
Comprehensive compiler correctness check:
- Validates observation/action/objective consistency
- Ensures IR executor works
- Creates Gymnasium environment successfully
- Inspector analysis functions
- Report: passed/failed with detailed error messages

## Integration: Full Pipeline

From Kobe program to trained policy:

```bash
# 1. Parse and compile
python -c "
from compiler import compile
from lexer import tokenize
from parser import Parser
from priorities import DEFAULTS

code = '''
policy { efficiency = 0.7; }
walk forward; stop;
'''
tokens = tokenize(code)
ast = Parser(tokens).parse()
ir = compile(ast, DEFAULTS)
print('Compiled to IR:', len(ir), 'instructions')
"

# 2. Generate environment
python -c "
from environment import create_kobe_environment
from backend import IRInterpreter
env = create_kobe_environment(ir, IRInterpreter, DEFAULTS)
obs, info = env.reset()
print('Generated environment, obs shape:', obs.shape)
"

# 3. Generate training script
python -c "
from trainer import generate_training_pipeline
generate_training_pipeline(ir, DEFAULTS, 'train.py')
print('Generated train.py')
"

# 4. Inspect what was generated
python -c "
from inspector import print_inspection_report
print_inspection_report(ir, DEFAULTS)
"

# 5. Run training
python train.py

# 6. Evaluate trained policy
python compare_baselines.py --trained --actor-path policy.pt
```

## Research Value

Each component directly supports research questions:

| Component | RQ | Purpose |
|-----------|----|---------| 
| environment.py | RQ5 | Semantic equivalence across simulation/hardware |
| trainer.py | RQ3 | Preserve understanding of objectives/learning |
| inspector.py | RQ4 | Does transparency improve debugging? |
| study_tasks.py | RQ1, RQ2 | Measure learning time, errors, workload |
| gate2_validator.py | RQ5 | Prove equivalence is achievable |

---

**Success Metrics:**
- ✓ All golden tests pass (10/10)
- ✓ Gate 2 validator passes (compiler proof)
- ✓ Generated env runs without error
- ✓ Training script executes successfully
- ✓ Study tasks are completable in estimated time

