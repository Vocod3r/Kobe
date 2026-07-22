from dataclasses import dataclass
import re
from typing import Iterator

@dataclass
class Token:
    type: str
    value: str
    line: int
    col: int
    
token_patterns = [
    ('COMMENT',    r'//[^\n]*'),          # must be before anything else
    ('NUMBER',     r'\d+(?:\.\d+)?'),     # 30, 0.5  — float before int would also work
    ('DOTDOT',     r'\.\.'),              # .. before any single dot
    ('COMPARATOR', r'<=|>=|==|!=|<|>'),  # two-char ops before one-char ops
    ('SEMICOLON',  r';'),
    ('EQUALS',     r'='),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('COMMA',      r','),
    ('WORD',       r'[A-Za-z_][A-Za-z0-9_]*'),  # catches everything alphabetic
    ('NEWLINE',    r'\n'),                # tracked separately for line counting
    ('SKIP',       r'[ \t\r]+'),          # whitespace — discarded
    ('UNKNOWN',    r'.'),                 # catch-all — becomes a lex error
]

master_pattern = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_patterns))

# All words that have special meaning in Kobe
KEYWORDS   = {'walk','run','turn','stop','wait','observe','if','loop',
               'until','break','then','else','policy','and','or','not','let'}
SENSORS    = {'dist','colour','IR','UV','touch','gyro','sound'}
POLICY_KEYS= {'curiosity','safety','comfort','efficiency'}
DIRECTIONS = {'forward','backward','left','right'}
SPEEDS     = {'slowly','normally','quickly'}
COLOURS    = {'red','orange','yellow','green','blue','indigo',
               'violet','white','black','none'}
UNITS      = {'cm','m','in','nm','deg','db','ms','sec'}
BOOLEANS   = {'detected','pressed','signal','index','tilt','is'}

def classify_word(value: str) -> str:
    if value in KEYWORDS: return 'KEYWORD'
    if value in SENSORS: return 'SENSOR'
    if value in POLICY_KEYS: return 'POLICY_KEY'
    if value in DIRECTIONS: return 'DIRECTION'
    if value in SPEEDS: return 'SPEED'
    if value in COLOURS: return 'COLOUR'
    if value in UNITS: return 'UNIT'
    if value in BOOLEANS: return value.upper()
    return 'IDENTIFIER'

class LexError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"line {line}, col {col}: {message}")
        self.line= line
        self.col = col
        
def tokenize(source: str) -> list[Token]:
    tokens = []
    line = 1
    line_start = 0
    for match in master_pattern.finditer(source):
        kind = match.lastgroup
        value = match.group()
        column = match.start() - line_start + 1
        
        if kind == 'NUMBER':
            value = float(value) if '.' in value else int(value)
            tokens.append(Token(kind, value, line, column))
        if kind == 'WORD':
            kind = classify_word(value)
            tokens.append(Token(kind, value, line, column))
        elif kind == 'NEWLINE':
            line += 1
            line_start = match.end()
        elif kind == 'SKIP' or kind == 'COMMENT':
            continue
        elif kind == 'UNKNOWN':
            raise LexError(f"Unknown token: {value}", line, column)
        else:
            tokens.append(Token(kind, value, line, column))
    return tokens

if __name__ == '__main__':
    source= """
    policy {safety = 0.9; 
    efficiency = 0.6;}
    walk forward slowly;
    
    loop until (dist < 20 cm) {
        observe (dist, colour){
            dist < 20 cm then {stop; break; }
            colour is red then {stop; break;}
        }
    }
    """
    for tok in tokenize(source):
        print(tok)
    