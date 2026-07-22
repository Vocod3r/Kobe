from lexer import Token, tokenize, LexError

class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"line {line}, col {col}: {message}")
        self.line = line
        self.col  = col

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos    = 0
        self._break_stack: list[list[int]] = []

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != 'EOF':
            self.pos += 1
        return tok

    def expect(self, type: str, value: str | None = None) -> Token:
        tok = self.peek()
        if tok.type != type:
            expected = f"'{value}'" if value else type
            raise ParseError(
                f"Expected {expected}, got '{tok.value}'", tok.line, tok.col
            )
        if value and tok.value != value:
            raise ParseError(
                f"Expected '{value}', got '{tok.value}'", tok.line, tok.col
            )
        return self.advance()

    def match(self, type: str, value: str | None = None) -> bool:
        tok = self.peek()
        if tok.type != type:
            return False
        if value and tok.value != value:
            return False
        return True

    def match_any(self, *pairs: tuple) -> bool:
        tok = self.peek()
        for t, v in pairs:
            if tok.type == t and (v is None or tok.value == v):
                return True
        return False

    def parse(self) -> dict:
        pos    = self.peek()
        policy = None

        if self.match('KEYWORD', 'policy'):
            policy = self.parse_policy()

        body = []
        while not self.match('EOF'):
            body.append(self.parse_statement())

        return {
            'type':   'Program',
            'policy': policy,
            'body':   body,
            'pos':    {'line': pos.line, 'col': pos.col}
        }

    def parse_policy(self) -> dict:
        tok = self.expect('KEYWORD', 'policy')
        self.expect('LBRACE')

        defaults = {
            'curiosity':  0.3,
            'safety':     0.5,
            'comfort':    0.5,
            'efficiency': 0.5
        }

        while not self.match('RBRACE'):
            if self.match('EOF'):
                raise ParseError("Unclosed policy block — missing '}'", tok.line, tok.col)
            key_tok = self.expect('POLICY_KEY')
            self.expect('EQUALS')
            val_tok = self.expect('NUMBER')
            self.expect('SEMICOLON')
            defaults[key_tok.value] = float(val_tok.value)

        self.expect('RBRACE')
        return {'type': 'Policy', **defaults, 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_statement(self) -> dict:
        tok = self.peek()

        if tok.type == 'KEYWORD':
            if tok.value in ('walk', 'run', 'turn', 'stop'):
                return self.parse_action()
            if tok.value == 'observe': return self.parse_observe()
            if tok.value == 'if':      return self.parse_if()
            if tok.value == 'loop':    return self.parse_loop()
            if tok.value == 'break':   return self.parse_break()
            if tok.value == 'wait':    return self.parse_wait()

        raise ParseError(
            f"Unexpected token '{tok.value}' — expected a statement",
            tok.line, tok.col
        )

    def parse_action(self) -> dict:
        tok = self.advance()

        if tok.value == 'stop':
            self.expect('SEMICOLON')
            return {'type': 'Stop', 'pos': {'line': tok.line, 'col': tok.col}}

        if tok.value == 'turn':
            if not self.match('DIRECTION'):
                raise ParseError("'turn' requires a direction: left or right", tok.line, tok.col)
            direction = self.advance().value
            self.expect('SEMICOLON')
            return {'type': 'Turn', 'direction': direction, 'pos': {'line': tok.line, 'col': tok.col}}

        direction = self.advance().value if self.match('DIRECTION') else None
        speed     = self.advance().value if self.match('SPEED')     else 'normally'
        self.expect('SEMICOLON')

        return {
            'type':      tok.value.capitalize(),
            'direction': direction,
            'speed':     speed,
            'pos':       {'line': tok.line, 'col': tok.col}
        }

    def parse_condition(self) -> dict:
        left = self.parse_and_expr()
        while self.match('KEYWORD', 'or'):
            self.advance()
            right = self.parse_and_expr()
            left  = {'type': 'Or', 'left': left, 'right': right}
        return left

    def parse_and_expr(self) -> dict:
        left = self.parse_not_expr()
        while self.match('KEYWORD', 'and'):
            self.advance()
            right = self.parse_not_expr()
            left  = {'type': 'And', 'left': left, 'right': right}
        return left

    def parse_not_expr(self) -> dict:
        if self.match('KEYWORD', 'not'):
            tok     = self.advance()
            operand = self.parse_not_expr()
            return {'type': 'Not', 'operand': operand, 'pos': {'line': tok.line, 'col': tok.col}}
        return self.parse_atomic_condition()

    def parse_atomic_condition(self) -> dict:
        tok = self.peek()

        if self.match('LPAREN'):
            self.advance()
            cond = self.parse_condition()
            self.expect('RPAREN')
            return cond

        if not self.match('SENSOR'):
            raise ParseError(
                f"Expected a condition (dist, colour, touch, ...) but got '{tok.value}'",
                tok.line, tok.col
            )

        sensor = self.advance().value

        if sensor == 'dist':   return self.parse_dist_condition(tok)
        if sensor == 'colour': return self.parse_colour_condition(tok)
        if sensor == 'touch':  return self.parse_touch_condition(tok)
        if sensor == 'IR':     return self.parse_ir_condition(tok)
        if sensor == 'UV':     return self.parse_uv_condition(tok)
        if sensor == 'gyro':   return self.parse_gyro_condition(tok)
        if sensor == 'sound':  return self.parse_sound_condition(tok)

        raise ParseError(f"Unknown sensor '{sensor}'", tok.line, tok.col)

    def parse_dist_condition(self, tok) -> dict:
        comparator = self.expect('COMPARATOR').value
        value      = float(self.expect('NUMBER').value)
        unit       = self.expect('UNIT').value
        return {'type': 'DistCondition', 'comparator': comparator,
                'value': value, 'unit': unit, 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_colour_condition(self, tok) -> dict:
        if self.match('IS'):
            self.advance()
            negate = False
        elif self.match('KEYWORD', 'not'):
            self.advance()
            negate = True
        else:
            raise ParseError("Expected 'is' or 'not' after 'colour'", tok.line, tok.col)

        colour = self._parse_colour_value(tok)
        return {'type': 'ColourCondition', 'negate': negate, 'colour': colour,
                'pos': {'line': tok.line, 'col': tok.col}}

    def _parse_colour_value(self, tok) -> dict | str:
        if self.match('COLOUR_VAL'):
            return self.advance().value
        nm_min = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'nm')
        self.expect('DOTDOT')
        nm_max = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'nm')
        return {'nmMin': nm_min, 'nmMax': nm_max}

    def parse_touch_condition(self, tok) -> dict:
        self.expect('PRESSED')
        return {'type': 'TouchCondition', 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_ir_condition(self, tok) -> dict:
        if self.match('DETECTED'):
            self.advance()
            return {'type': 'IRCondition', 'mode': 'detected',
                    'pos': {'line': tok.line, 'col': tok.col}}
        self.expect('SIGNAL')
        comparator = self.expect('COMPARATOR').value
        value      = float(self.expect('NUMBER').value)
        return {'type': 'IRCondition', 'mode': 'signal',
                'comparator': comparator, 'value': value,
                'pos': {'line': tok.line, 'col': tok.col}}

    def parse_uv_condition(self, tok) -> dict:
        if self.match('DETECTED'):
            self.advance()
            return {'type': 'UVCondition', 'mode': 'detected',
                    'pos': {'line': tok.line, 'col': tok.col}}
        self.expect('INDEX')
        comparator = self.expect('COMPARATOR').value
        index      = float(self.expect('NUMBER').value)
        return {'type': 'UVCondition', 'mode': 'index',
                'comparator': comparator, 'index': index,
                'pos': {'line': tok.line, 'col': tok.col}}

    def parse_gyro_condition(self, tok) -> dict:
        self.expect('TILT')
        comparator = self.expect('COMPARATOR').value
        degrees    = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'deg')
        return {'type': 'GyroCondition', 'comparator': comparator,
                'degrees': degrees, 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_sound_condition(self, tok) -> dict:
        comparator = self.expect('COMPARATOR').value
        db         = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'db')
        return {'type': 'SoundCondition', 'comparator': comparator,
                'db': db, 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_observe(self) -> dict:
        tok = self.expect('KEYWORD', 'observe')
        self.expect('LPAREN')

        sensors = [self.expect('SENSOR').value]
        while self.match('COMMA'):
            self.advance()
            sensors.append(self.expect('SENSOR').value)

        self.expect('RPAREN')
        self.expect('LBRACE')

        branches = []
        while not self.match('RBRACE'):
            if self.match('EOF'):
                raise ParseError("Unclosed observe block — missing '}'", tok.line, tok.col)
            branches.append(self.parse_sensor_branch())

        self.expect('RBRACE')
        return {'type': 'Observe', 'sensors': sensors, 'branches': branches,
                'pos': {'line': tok.line, 'col': tok.col}}

    def parse_sensor_branch(self) -> dict:
        condition = self.parse_condition()
        self.expect('KEYWORD', 'then')
        then_body = self.parse_block()
        else_body = []
        if self.match('KEYWORD', 'else'):
            self.advance()
            else_body = self.parse_block()
        return {'type': 'SensorBranch', 'condition': condition,
                'then': then_body, 'else': else_body}

    def parse_if(self) -> dict:
        tok       = self.expect('KEYWORD', 'if')
        condition = self.parse_condition()
        self.expect('KEYWORD', 'then')
        then_body = self.parse_block()
        else_body = []
        if self.match('KEYWORD', 'else'):
            self.advance()
            else_body = self.parse_block()
        return {'type': 'If', 'condition': condition,
                'then': then_body, 'else': else_body,
                'pos': {'line': tok.line, 'col': tok.col}}

    def parse_loop(self) -> dict:
        tok = self.expect('KEYWORD', 'loop')

        if self.match('KEYWORD', 'until'):
            self.advance()
            self.expect('LPAREN')
            condition = self.parse_condition()
            self.expect('RPAREN')
            body = self.parse_block()
            return {'type': 'LoopUntil', 'condition': condition,
                    'body': body, 'pos': {'line': tok.line, 'col': tok.col}}

        self.expect('LPAREN')
        count = self.expect('NUMBER')
        self.expect('RPAREN')
        body  = self.parse_block()
        return {'type': 'LoopFor', 'count': float(count.value),
                'body': body, 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_wait(self) -> dict:
        tok      = self.expect('KEYWORD', 'wait')
        duration = float(self.expect('NUMBER').value)
        unit     = self.expect('UNIT').value
        self.expect('SEMICOLON')
        return {'type': 'Wait', 'duration': duration, 'unit': unit,
                'pos': {'line': tok.line, 'col': tok.col}}

    def parse_break(self) -> dict:
        tok = self.expect('KEYWORD', 'break')
        self.expect('SEMICOLON')
        return {'type': 'Break', 'pos': {'line': tok.line, 'col': tok.col}}

    def parse_block(self) -> list:
        self.expect('LBRACE')
        body = []
        while not self.match('RBRACE'):
            if self.match('EOF'):
                raise ParseError("Unclosed block — missing '}'",
                                 self.peek().line, self.peek().col)
            body.append(self.parse_statement())
        self.expect('RBRACE')
        return body


# ── Module-level entry point ─────────────────────────────────────

def parse(source: str) -> dict:
    tokens = tokenize(source)
    return Parser(tokens).parse()


if __name__ == '__main__':
    import json

    source = """
    policy {
        safety     = 0.9;
        efficiency = 0.6;
    }

    walk forward slowly;

    loop until (dist < 20 cm) {
        observe(dist, colour) {
            dist < 20 cm then { stop; break; }
            colour is red then { stop; break; }
        }
    }

    turn left;
    walk forward;
    """

    ast = parse(source)
    print(json.dumps(ast, indent=2))