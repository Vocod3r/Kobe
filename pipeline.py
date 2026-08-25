# pipeline.py
from lexer import tokenize
from parser import Parser
from semantic_analyzer import analyze, blocks_training
from priorities import extract_priorities
from compiler import compile
from codegen import generate, slider_descriptions


def _algorithm_from_ast(ast: dict) -> str:
    if ast.get('algorithm'):
        return ast['algorithm']['name']
    return 'SAC'


def run(source: str, trial_level: int = 2, target: str = 'rl') -> dict:
    """
    Full Kobe pipeline.
    Returns dict with diagnostics, IR, priorities, and generated code.
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

    return result