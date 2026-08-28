# compiler.py

def compile(ast: dict, priorities: dict) -> list[dict]:
    c = Compiler(priorities)
    c.compile_program(ast)
    return c.ir

class Compiler:
    def __init__(self, priorities: dict):
        self.ir          = []
        self.priorities  = priorities
        self._break_stack: list[list[int]] = []

    def emit(self, instruction: dict):
        self.ir.append(instruction)
        return len(self.ir) - 1   # returns index of emitted instruction

    def patch(self, index: int, key: str, value: int):
        """Backpatch a jump target once we know where it points."""
        self.ir[index][key] = value

    def current_index(self) -> int:
        return len(self.ir)

    # --- Program ---

    def compile_program(self, ast: dict):
        if ast.get('algorithm'):
            self.emit({'op': 'ALGORITHM', 'name': ast['algorithm']['name']})

        if ast.get('hardware'):
            hw = ast['hardware']
            self.emit({
                'op':     'HARDWARE',
                'target': hw.get('target', 'EV3'),
                'motors': hw.get('motors', []),
                'sensors': hw.get('sensors', []),
            })

        self.emit({
            'op':         'POLICY',
            'curiosity':  self.priorities['curiosity'],
            'safety':     self.priorities['safety'],
            'comfort':    self.priorities['comfort'],
            'efficiency': self.priorities['efficiency']
        })
        for stmt in ast['body']:
            self.compile_statement(stmt)
        self.emit({'op': 'HALT'})

    # --- Statements ---

    def compile_statement(self, node: dict):
        t = node['type']
        if t == 'Walk':         self.compile_walk(node)
        elif t == 'Run':        self.compile_run(node)
        elif t == 'Turn':       self.emit({'op': 'TURN', 'direction': node['direction']})
        elif t == 'Stop':       self.emit({'op': 'STOP'})
        elif t == 'Wait':       self.compile_wait(node)
        elif t == 'Observe':    self.compile_observe(node)
        elif t == 'If':         self.compile_if(node)
        elif t == 'LoopFor':    self.compile_loop_for(node)
        elif t == 'LoopUntil':  self.compile_loop_until(node)
        elif t == 'Break':      self.compile_break(node)

    def compile_walk(self, node: dict):
        self.emit({
            'op':             'WALK',
            'direction':      node.get('direction') or 'forward',
            'speedMultiplier': _speed_multiplier(node.get('speed', 'normally'))
        })

    def compile_run(self, node: dict):
        self.emit({
            'op':             'RUN',
            'direction':      node.get('direction') or 'forward',
            'speedMultiplier': _speed_multiplier(node.get('speed', 'normally')) * 2
        })

    def compile_wait(self, node: dict):
        duration_ms = node['duration'] * 1000 if node['unit'] == 'sec' else node['duration']
        self.emit({'op': 'WAIT', 'durationMs': duration_ms})

    # --- Observe ---

    def compile_observe(self, node: dict):
        self.emit({'op': 'SENSE', 'sensors': node['sensors']})

        for branch in node['branches']:
            self.compile_condition(branch['condition'])

            # Emit conditional jump — target unknown yet
            jump_idx = self.emit({'op': 'JUMP_IF_FALSE', 'target': -1})

            for stmt in branch['then']:
                self.compile_statement(stmt)

            if branch['else']:
                # Jump over else block after then block
                else_skip_idx = self.emit({'op': 'JUMP', 'target': -1})
                self.patch(jump_idx, 'target', self.current_index())
                for stmt in branch['else']:
                    self.compile_statement(stmt)
                self.patch(else_skip_idx, 'target', self.current_index())
            else:
                self.patch(jump_idx, 'target', self.current_index())

    # --- If ---

    def compile_if(self, node: dict):
        self.compile_condition(node['condition'])
        jump_idx = self.emit({'op': 'JUMP_IF_FALSE', 'target': -1})

        for stmt in node['then']:
            self.compile_statement(stmt)

        if node['else']:
            else_skip_idx = self.emit({'op': 'JUMP', 'target': -1})
            self.patch(jump_idx, 'target', self.current_index())
            for stmt in node['else']:
                self.compile_statement(stmt)
            self.patch(else_skip_idx, 'target', self.current_index())
        else:
            self.patch(jump_idx, 'target', self.current_index())

    # --- Loops ---

    def compile_loop_for(self, node: dict):
        loop_start = self.current_index()
        self.emit({'op': 'LOOP_START', 'count': int(node['count'])})

        # Track break sites to backpatch
        break_sites = self._compile_body_with_breaks(node['body'])

        end_idx = self.emit({'op': 'LOOP_END', 'startTarget': loop_start})

        # Patch all breaks to jump past LOOP_END
        exit_target = self.current_index()
        for site in break_sites:
            self.patch(site, 'target', exit_target)

    def compile_loop_until(self, node: dict):
        cond_start = self.current_index()

        self.compile_condition(node['condition'])
        exit_jump = self.emit({'op': 'JUMP_IF_FALSE', 'target': -1})

        break_sites = self._compile_body_with_breaks(node['body'])

        self.emit({'op': 'JUMP', 'target': cond_start})

        exit_target = self.current_index()
        self.patch(exit_jump, 'target', exit_target)
        for site in break_sites:
            self.patch(site, 'target', exit_target)

    def _compile_body_with_breaks(self, body: list) -> list[int]:
        """Compile a loop body. Returns list of JUMP indices emitted by break statements."""
        self._break_stack.append([])
        for stmt in body:
            self.compile_statement(stmt)
        return self._break_stack.pop()

    def compile_break(self, node: dict):
        idx = self.emit({'op': 'JUMP', 'target': -1})   # target patched later
        if self._break_stack:
            self._break_stack[-1].append(idx)

    # --- Conditions ---

    def compile_condition(self, cond: dict):
        t = cond['type']

        if t == 'And':
            self.compile_condition(cond['left'])
            self.compile_condition(cond['right'])
            self.emit({'op': 'AND'})

        elif t == 'Or':
            self.compile_condition(cond['left'])
            self.compile_condition(cond['right'])
            self.emit({'op': 'OR'})

        elif t == 'Not':
            self.compile_condition(cond['operand'])
            self.emit({'op': 'NOT'})

        elif t == 'DistCondition':
            cm = _to_cm(cond['value'], cond['unit'])
            self.emit({'op': 'CMP_DIST', 'comparator': cond['comparator'], 'valueCm': cm})

        elif t == 'ColourCondition':
            self.emit({'op': 'CMP_COLOUR', 'negate': cond['negate'], 'colour': cond['colour']})

        elif t == 'TouchCondition':
            self.emit({'op': 'CMP_TOUCH'})

        elif t == 'IRCondition':
            self.emit({'op': 'CMP_IR', 'mode': cond['mode'],
                       'comparator': cond.get('comparator'),
                       'value': cond.get('value')})

        elif t == 'UVCondition':
            self.emit({'op': 'CMP_UV', 'mode': cond['mode'],
                       'comparator': cond.get('comparator'),
                       'index': cond.get('index')})

        elif t == 'GyroCondition':
            self.emit({'op': 'CMP_GYRO', 'comparator': cond['comparator'],
                       'degrees': cond['degrees']})

        elif t == 'SoundCondition':
            self.emit({'op': 'CMP_SOUND', 'comparator': cond['comparator'],
                       'db': cond['db']})


# --- Helpers ---

def _speed_multiplier(speed: str) -> float:
    return {'slowly': 0.5, 'normally': 1.0, 'quickly': 1.5}.get(speed, 1.0)

def _to_cm(value: float, unit: str) -> float:
    if unit == 'm':  return value * 100
    if unit == 'in': return value * 2.54
    return value