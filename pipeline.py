# pipeline.py
from lexer    import tokenize
from parser   import Parser
from semantic_analyzer import analyze, blocks_training
from priorities import extract_priorities
from compiler import compile
from codegen import generate_ev3
from codegen import generate_rpi

def run(source: str, trial_level: int = 2, target: str = 'rl') -> dict:
    """
    Full Kobe pipeline.
    Returns dict with diagnostics, IR, priorities, and generated code.
    """
    # 1. Lex + Parse
    tokens = tokenize(source)
    ast    = Parser(tokens).parse()

    # 2. Semantic analysis
    diagnostics = analyze(ast)

    result = {
        'diagnostics': [
            {'severity': d.severity, 'message': d.message,
             'line': d.line, 'col': d.col}
            for d in diagnostics
        ]
    }

    # 3. Block if errors or warnings
    if blocks_training(diagnostics):
        result['blocked'] = True
        return result

    result['blocked'] = False

    # 4. Extract priorities
    priorities = extract_priorities(ast)
    result['priorities'] = priorities

    # 5. Compile to IR
    ir = compile(ast, priorities)
    result['ir'] = ir

    # 6. Generate target code
    if target == 'ev3':
        result['code'] = generate_ev3(ir)
    elif target == 'rpi':
        result['code'] = generate_rpi(ir)

    return result