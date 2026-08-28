# pipeline.py
from lexer import tokenize
from parser import Parser
from semantic_analyzer import analyze, blocks_training
from priorities import extract_priorities
from compiler import compile
from codegen import generate, slider_descriptions
from gate2_validator import validate_gate_2


def _algorithm_from_ast(ast: dict) -> str:
    if ast.get('algorithm'):
        return ast['algorithm']['name']
    return 'SAC'


def run(source: str, trial_level: int = 2, target: str = 'rl', validate: bool = True) -> dict:
    """
    Full Kobe pipeline.
    Returns dict with diagnostics, IR, priorities, and generated code.

    If `validate` is True (default), every clean compile is also run through
    Gate 2 (gate2_validator.validate_gate_2): the reference interpreter and
    the IR trace executor are independently run against the same scripted
    sensor scenario, and their traces must match exactly. This is the "single
    source of truth" guarantee — the IDE never shows IR/generated code that
    hasn't been checked against source-level semantics.
    """
    tokens = tokenize(source)
    ast = Parser(tokens).parse()

    diagnostics = analyze(ast)
    algorithm = _algorithm_from_ast(ast)
    hardware = ast.get('hardware')

    result = {
        'diagnostics': [
            {
                'severity': d.severity,
                'message': d.message,
                'line': d.line,
                'col': d.col,
            }
            for d in diagnostics
        ],
        'algorithm': algorithm,
        'hardware': hardware,
        'sliderDescriptions': slider_descriptions(algorithm),
    }

    if blocks_training(diagnostics):
        result['blocked'] = True
        return result

    result['blocked'] = False
    priorities = extract_priorities(ast)
    result['priorities'] = priorities

    ir = compile(ast, priorities)
    result['ir'] = ir

    if target in ('rl', 'ev3', 'rpi'):
        result['code'] = generate(ir, priorities, hardware, algorithm)

    if validate:
        try:
            gate2 = validate_gate_2(ir, priorities, ast=ast)
            result['gate2'] = {
                'passed': gate2['passed'],
                'checks': gate2['checks'],
                'errors': gate2['errors'],
            }
        except Exception as e:
            result['gate2'] = {
                'passed': False,
                'checks': {},
                'errors': [f'Gate 2 validator crashed: {e}'],
            }

    return result