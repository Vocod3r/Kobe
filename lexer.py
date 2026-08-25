from dataclasses import dataclass
import re

# ── Token ─────────────────────────────────────────────────────────────────────

@dataclass
class Token:
    type:  str
    value: str    # always str — parser does numeric conversion
    line:  int
    col:   int

# ── Patterns — most specific first ───────────────────────────────────────────

_PATTERNS = [
    ('COMMENT',    r'//[^\n]*'),
    ('NUMBER',     r'\d+(?:\.\d+)?'),
    ('DOTDOT',     r'\.\.'),
    ('COMPARATOR', r'<=|>=|==|!=|<|>'),
    ('SEMICOLON',  r';'),
    ('COLON',      r':'),
    ('AT',         r'@'),
    ('EQUALS',     r'='),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('LBRACKET',   r'\['),
    ('RBRACKET',   r'\]'),
    ('COMMA',      r','),
    ('WORD',       r'[A-Za-z_][A-Za-z0-9_]*'),
    ('NEWLINE',    r'\n'),
    ('SKIP',       r'[ \t\r]+'),
    ('UNKNOWN',    r'.'),
]

_MASTER = re.compile(
    '|'.join(f'(?P<{name}>{pat})' for name, pat in _PATTERNS)
)

# ── Word classification ───────────────────────────────────────────────────────
#
# EXACT_TOKENS: each word gets its own token type so the parser can call
# self.expect('DETECTED') / self.match('IS') unambiguously.
# Never collapse these into a generic type.

KEYWORDS       = {'walk','run','turn','stop','wait','observe','if','loop',
                   'until','break','then','else','policy','and','or','not',
                   'algorithm','hardware'}

SENSORS        = {'dist','colour','IR','UV','touch','gyro','sound'}
POLICY_KEYS    = {'curiosity','safety','comfort','efficiency'}
ALGORITHM_VALS = {'SAC','TD3','DroQ','random'}
HW_TARGETS     = {'EV3','Spike','RPi','RaspberryPi'}
HW_KEYS        = {'target','motors','sensors'}
DIRECTIONS     = {'forward','backward','left','right'}
SPEEDS         = {'slowly','normally','quickly'}
COLOURS        = {'red','orange','yellow','green','blue','indigo',
                   'violet','white','black','none'}
UNITS          = {'cm','m','in','nm','deg','db','ms','sec'}

EXACT_TOKENS = {
    'detected': 'DETECTED',
    'pressed':  'PRESSED',
    'signal':   'SIGNAL',
    'index':    'INDEX',
    'tilt':     'TILT',
    'is':       'IS',
}


def classify_word(value: str) -> str:
    if value in KEYWORDS:       return 'KEYWORD'
    if value in EXACT_TOKENS:   return EXACT_TOKENS[value]
    if value in SENSORS:        return 'SENSOR'
    if value in POLICY_KEYS:    return 'POLICY_KEY'
    if value in ALGORITHM_VALS: return 'ALGORITHM_VAL'
    if value in HW_TARGETS:     return 'HW_TARGET'
    if value in HW_KEYS:        return 'HW_KEY'
    if value in DIRECTIONS:     return 'DIRECTION'
    if value in SPEEDS:         return 'SPEED'
    if value in COLOURS:        return 'COLOUR_VAL'
    if value in UNITS:          return 'UNIT'
    return 'IDENTIFIER'


# ── Error ─────────────────────────────────────────────────────────────────────

class LexError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"line {line}, col {col}: {message}")
        self.line = line
        self.col  = col


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize(source: str) -> list[Token]:
    tokens     = []
    line       = 1
    line_start = 0

    for match in _MASTER.finditer(source):
        kind  = match.lastgroup
        value = match.group()
        col   = match.start() - line_start + 1

        if kind == 'NEWLINE':
            line += 1; line_start = match.end(); continue
        if kind in ('SKIP', 'COMMENT'):
            continue
        if kind == 'UNKNOWN':
            raise LexError(f"Unexpected character '{value}'", line, col)
        if kind == 'WORD':
            kind = classify_word(value)

        tokens.append(Token(kind, value, line, col))

    tokens.append(Token('EOF', '', line, len(source) - line_start + 1))
    return tokens


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    source = """
    algorithm DroQ

    hardware {
        target: EV3
        motors: [A, B]
        sensors: [dist@1, colour@2]
    }

    policy { safety = 0.9; efficiency = 0.6; }

    walk forward slowly;

    loop until (dist < 20 cm) {
        observe(dist, colour) {
            dist < 20 cm then { stop; break; }
            colour is red then { stop; break; }
            IR detected then { stop; }
            touch pressed then { stop; }
            gyro tilt > 30 deg then { stop; }
        }
    }

    turn left;
    walk forward;
    """

    for tok in tokenize(source):
        print(f"{tok.type:<16} {repr(tok.value):<20} line {tok.line} col {tok.col}")