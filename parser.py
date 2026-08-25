from lexer import Token, tokenize, LexError


class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"line {line}, col {col}: {message}")
        self.line = line
        self.col  = col


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens      = tokens
        self.pos         = 0
        self._break_stack: list[list[int]] = []

    # ── Infrastructure ────────────────────────────────────────────────────────

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
            raise ParseError(f"Expected {expected}, got '{tok.value}'", tok.line, tok.col)
        if value and tok.value != value:
            raise ParseError(f"Expected '{value}', got '{tok.value}'", tok.line, tok.col)
        return self.advance()

    def match(self, type: str, value: str | None = None) -> bool:
        tok = self.peek()
        if tok.type != type: return False
        if value and tok.value != value: return False
        return True

    def match_any(self, *pairs: tuple) -> bool:
        tok = self.peek()
        for t, v in pairs:
            if tok.type == t and (v is None or tok.value == v):
                return True
        return False

    # ── Program ───────────────────────────────────────────────────────────────
    #
    # Grammar:
    #   program ::= algorithm_stmt? hardware_block? policy_block? statement*
    #
    # All three top-level declarations are optional and must appear in this
    # order if present. Statements follow.

    def parse(self) -> dict:
        pos = self.peek()

        algorithm = None
        if self.match('KEYWORD', 'algorithm'):
            algorithm = self.parse_algorithm()

        hardware = None
        if self.match('KEYWORD', 'hardware'):
            hardware = self.parse_hardware()

        policy = None
        if self.match('KEYWORD', 'policy'):
            policy = self.parse_policy()

        body = []
        while not self.match('EOF'):
            body.append(self.parse_statement())

        return {
            'type':      'Program',
            'algorithm': algorithm,
            'hardware':  hardware,
            'policy':    policy,
            'body':      body,
            'pos':       {'line': pos.line, 'col': pos.col},
        }

    # ── Algorithm block ───────────────────────────────────────────────────────
    #
    # algorithm_stmt ::= "algorithm" ALGORITHM_VAL
    #
    # One line, no braces, no semicolon.
    # Valid values: SAC | TD3 | DroQ | random

    def parse_algorithm(self) -> dict:
        tok = self.expect('KEYWORD', 'algorithm')
        if not self.match('ALGORITHM_VAL'):
            raise ParseError(
                f"Expected algorithm name (SAC, TD3, DroQ, random), got '{self.peek().value}'",
                self.peek().line, self.peek().col
            )
        name = self.advance()
        return {
            'type':  'Algorithm',
            'name':  name.value,
            'pos':   {'line': tok.line, 'col': tok.col},
        }

    # ── Hardware block ────────────────────────────────────────────────────────
    #
    # hardware_block ::= "hardware" "{" hw_setting* "}"
    #
    # hw_setting ::= "target"  ":" HW_TARGET
    #              | "motors"  ":" "[" port ("," port)* "]"
    #              | "sensors" ":" "[" sensor_decl ("," sensor_decl)* "]"
    #
    # sensor_decl ::= SENSOR "@" port
    # port        ::= IDENTIFIER | NUMBER   (e.g. A, B, 1, 2)

    def parse_hardware(self) -> dict:
        tok = self.expect('KEYWORD', 'hardware')
        self.expect('LBRACE')

        result = {
            'type':    'Hardware',
            'target':  'EV3',
            'motors':  [],
            'sensors': [],
            'pos':     {'line': tok.line, 'col': tok.col},
        }

        while not self.match('RBRACE'):
            if self.match('EOF'):
                raise ParseError("Unclosed hardware block — missing '}'", tok.line, tok.col)

            if not self.match('HW_KEY'):
                raise ParseError(
                    f"Expected hardware setting (target, motors, sensors), got '{self.peek().value}'",
                    self.peek().line, self.peek().col
                )

            key = self.advance().value
            self.expect('COLON')

            if key == 'target':
                if not self.match('HW_TARGET'):
                    raise ParseError(
                        f"Expected hardware target (EV3, Spike, RPi), got '{self.peek().value}'",
                        self.peek().line, self.peek().col
                    )
                result['target'] = self.advance().value

            elif key == 'motors':
                result['motors'] = self._parse_port_list()

            elif key == 'sensors':
                result['sensors'] = self._parse_sensor_decl_list()

        self.expect('RBRACE')
        return result

    def _parse_port_list(self) -> list[dict]:
        """[ A, B ] or [ A, B, C, D ]"""
        self.expect('LBRACKET')
        ports = [{'port': self._parse_port()}]
        while self.match('COMMA'):
            self.advance()
            ports.append({'port': self._parse_port()})
        self.expect('RBRACKET')
        return ports

    def _parse_sensor_decl_list(self) -> list[dict]:
        """[ dist@1, colour@2 ]"""
        self.expect('LBRACKET')
        decls = [self._parse_sensor_decl()]
        while self.match('COMMA'):
            self.advance()
            decls.append(self._parse_sensor_decl())
        self.expect('RBRACKET')
        return decls

    def _parse_sensor_decl(self) -> dict:
        """dist@1"""
        if not self.match('SENSOR'):
            raise ParseError(
                f"Expected sensor name (dist, colour, IR...), got '{self.peek().value}'",
                self.peek().line, self.peek().col
            )
        sensor = self.advance().value
        self.expect('AT')
        port = self._parse_port()
        return {'type': sensor, 'port': port}

    def _parse_port(self) -> str:
        """Port is either an IDENTIFIER (A, B) or a NUMBER (1, 2)."""
        tok = self.peek()
        if tok.type in ('IDENTIFIER', 'NUMBER', 'ALGORITHM_VAL', 'HW_TARGET',
                        'DIRECTION', 'COLOUR_VAL'):
            # Accept any single-token value as a port name
            return self.advance().value
        raise ParseError(
            f"Expected port name (A, B, 1, 2...), got '{tok.value}'",
            tok.line, tok.col
        )

    # ── Policy block ──────────────────────────────────────────────────────────

    def parse_policy(self) -> dict:
        tok = self.expect('KEYWORD', 'policy')
        self.expect('LBRACE')

        defaults = {'curiosity': 0.3, 'safety': 0.5, 'comfort': 0.5, 'efficiency': 0.5}

        while not self.match('RBRACE'):
            if self.match('EOF'):
                raise ParseError("Unclosed policy block — missing '}'", tok.line, tok.col)
            key = self.expect('POLICY_KEY')
            self.expect('EQUALS')
            val = self.expect('NUMBER')
            self.expect('SEMICOLON')
            defaults[key.value] = float(val.value)

        self.expect('RBRACE')
        return {'type': 'Policy', **defaults, 'pos': {'line': tok.line, 'col': tok.col}}

    # ── Statement dispatcher ──────────────────────────────────────────────────

    def parse_statement(self) -> dict:
        tok = self.peek()
        if tok.type == 'KEYWORD':
            if tok.value in ('walk', 'run', 'turn', 'stop'): return self.parse_action()
            if tok.value == 'observe': return self.parse_observe()
            if tok.value == 'if':     return self.parse_if()
            if tok.value == 'loop':   return self.parse_loop()
            if tok.value == 'break':  return self.parse_break()
            if tok.value == 'wait':   return self.parse_wait()

        raise ParseError(
            f"Unexpected token '{tok.value}' — expected a statement",
            tok.line, tok.col
        )

    # ── Actions ───────────────────────────────────────────────────────────────

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
            'pos':       {'line': tok.line, 'col': tok.col},
        }

    # ── Conditions ────────────────────────────────────────────────────────────
    #
    # Precedence (low → high): or → and → not → atomic
    # Each level calls the one below it.

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
                f"Expected condition (dist, colour, touch...) got '{tok.value}'",
                tok.line, tok.col
            )

        sensor = self.advance().value

        if sensor == 'dist':   return self._dist_cond(tok)
        if sensor == 'colour': return self._colour_cond(tok)
        if sensor == 'touch':  return self._touch_cond(tok)
        if sensor == 'IR':     return self._ir_cond(tok)
        if sensor == 'UV':     return self._uv_cond(tok)
        if sensor == 'gyro':   return self._gyro_cond(tok)
        if sensor == 'sound':  return self._sound_cond(tok)

        raise ParseError(f"Unknown sensor '{sensor}'", tok.line, tok.col)

    def _dist_cond(self, tok) -> dict:
        cmp   = self.expect('COMPARATOR').value
        val   = float(self.expect('NUMBER').value)
        unit  = self.expect('UNIT').value
        return {'type': 'DistCondition', 'comparator': cmp,
                'value': val, 'unit': unit, 'pos': {'line': tok.line, 'col': tok.col}}

    def _colour_cond(self, tok) -> dict:
        if self.match('IS'):
            self.advance(); negate = False
        elif self.match('KEYWORD', 'not'):
            self.advance(); negate = True
        else:
            raise ParseError("Expected 'is' or 'not' after 'colour'", tok.line, tok.col)

        colour = (self.advance().value if self.match('COLOUR_VAL')
                  else self._wavelength_range(tok))
        return {'type': 'ColourCondition', 'negate': negate, 'colour': colour,
                'pos': {'line': tok.line, 'col': tok.col}}

    def _wavelength_range(self, tok) -> dict:
        lo = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'nm')
        self.expect('DOTDOT')
        hi = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'nm')
        return {'nmMin': lo, 'nmMax': hi}

    def _touch_cond(self, tok) -> dict:
        self.expect('PRESSED')
        return {'type': 'TouchCondition', 'pos': {'line': tok.line, 'col': tok.col}}

    def _ir_cond(self, tok) -> dict:
        if self.match('DETECTED'):
            self.advance()
            return {'type': 'IRCondition', 'mode': 'detected',
                    'pos': {'line': tok.line, 'col': tok.col}}
        self.expect('SIGNAL')
        cmp = self.expect('COMPARATOR').value
        val = float(self.expect('NUMBER').value)
        return {'type': 'IRCondition', 'mode': 'signal',
                'comparator': cmp, 'value': val, 'pos': {'line': tok.line, 'col': tok.col}}

    def _uv_cond(self, tok) -> dict:
        if self.match('DETECTED'):
            self.advance()
            return {'type': 'UVCondition', 'mode': 'detected',
                    'pos': {'line': tok.line, 'col': tok.col}}
        self.expect('INDEX')
        cmp   = self.expect('COMPARATOR').value
        index = float(self.expect('NUMBER').value)
        return {'type': 'UVCondition', 'mode': 'index',
                'comparator': cmp, 'index': index, 'pos': {'line': tok.line, 'col': tok.col}}

    def _gyro_cond(self, tok) -> dict:
        self.expect('TILT')
        cmp     = self.expect('COMPARATOR').value
        degrees = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'deg')
        return {'type': 'GyroCondition', 'comparator': cmp,
                'degrees': degrees, 'pos': {'line': tok.line, 'col': tok.col}}

    def _sound_cond(self, tok) -> dict:
        cmp = self.expect('COMPARATOR').value
        db  = float(self.expect('NUMBER').value)
        self.expect('UNIT', 'db')
        return {'type': 'SoundCondition', 'comparator': cmp,
                'db': db, 'pos': {'line': tok.line, 'col': tok.col}}

    # ── Observe ───────────────────────────────────────────────────────────────

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
            branches.append(self._sensor_branch())

        self.expect('RBRACE')
        return {'type': 'Observe', 'sensors': sensors, 'branches': branches,
                'pos': {'line': tok.line, 'col': tok.col}}

    def _sensor_branch(self) -> dict:
        condition = self.parse_condition()
        self.expect('KEYWORD', 'then')
        then_body = self.parse_block()
        else_body = []
        if self.match('KEYWORD', 'else'):
            self.advance()
            else_body = self.parse_block()
        return {'type': 'SensorBranch', 'condition': condition,
                'then': then_body, 'else': else_body}

    # ── If ────────────────────────────────────────────────────────────────────

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

    # ── Loops ─────────────────────────────────────────────────────────────────

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
        count = float(self.expect('NUMBER').value)
        self.expect('RPAREN')
        body  = self.parse_block()
        return {'type': 'LoopFor', 'count': count,
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


# ── Public entry point ────────────────────────────────────────────────────────

def parse(source: str) -> dict:
    tokens = tokenize(source)
    return Parser(tokens).parse()


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json

    source = """
    algorithm DroQ

    hardware {
        target: EV3
        motors: [A, B]
        sensors: [dist@1, colour@2]
    }

    policy {
        safety     = 0.9;
        efficiency = 0.6;
    }

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

    ast = parse(source)
    print(json.dumps(ast, indent=2))