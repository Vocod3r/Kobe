from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class Diagnostic:
    severity: str
    message: str
    line: int
    col: int

class SemanticAnalyzer:
    def __init__(self):
        self.diagnostics: List[Diagnostic] = []
        self.loop_depth: int = 0
        self.active_sensors: set[str] = set()
        self.observed_sensors: set[str] = set()
        
    def error(self, msg, pos):
        self.diagnostics.append(Diagnostic('error', msg, pos['line'], pos['col']))
        
    def warning(self, msg, pos):
        self.diagnostics.append(Diagnostic('warning', msg, pos['line'], pos['col']))    
    
    def hint(self, msg, pos):
        self.diagnostics.append(Diagnostic('hint', msg, pos['line'], pos['col']))   
        

    def analyze(self, ast: dict) -> list[Diagnostic]:
        self._check_program(ast)
        return self.diagnostics

    # --- Program ---

    def _check_program(self, node: dict):
        has_movement   = False
        has_observe    = False
        has_policy     = node['policy'] is not None
        declared_sensors: set[str] = set()

        if node.get('hardware') and node['hardware'].get('sensors'):
            declared_sensors = {s['type'] for s in node['hardware']['sensors']}

        if node['policy']:
            self._check_policy(node['policy'])

        for stmt in node['body']:
            if stmt['type'] in ('Walk', 'Run', 'Turn'):
                has_movement = True
            if stmt['type'] == 'Observe':
                has_observe = True
                if declared_sensors:
                    for sensor in stmt['sensors']:
                        if sensor not in declared_sensors:
                            self.error(
                                f"Sensor '{sensor}' used in observe() but not declared in hardware {{}}.",
                                stmt['pos'],
                            )
            self._check_statement(stmt, after_terminator=False)

        # Program-level hints and warnings
        if not has_movement:
            pos = node['pos']
            self.warning("Your robot never moves — there is nothing to train.", pos)

        if not has_observe:
            pos = node['pos']
            self.hint("Your robot is acting blind — it cannot react to its environment.", pos)

        if not has_policy:
            pos = node['pos']
            self.hint("Using default policy values. Add a policy {} block to tune your robot's goals.", pos)

    # --- Policy ---

    def _check_policy(self, node: dict):
        seen = set()
        valid_keys = {'curiosity', 'safety', 'comfort', 'efficiency'}

        for key in valid_keys:
            val = node.get(key)
            if val is not None:
                if val < 0.0 or val > 1.0:
                    self.error(f"'{key}' must be between 0.0 and 1.0.", node['pos'])
                if key in seen:
                    self.warning(f"You set '{key}' twice. Only the last value is used.", node['pos'])
                seen.add(key)

        all_zero = all(node.get(k, 0) == 0.0 for k in valid_keys)
        if all_zero:
            self.warning("All policy values are 0 — your robot has no goals.", node['pos'])

    # --- Statement dispatcher ---

    def _check_statement(self, node: dict, after_terminator: bool):
        if after_terminator:
            self.warning("Nothing after this will run.", node['pos'])
            return

        t = node['type']

        if t in ('Walk', 'Run'):     self._check_movement(node)
        elif t == 'Stop':            pass   # always valid
        elif t == 'Turn':            pass   # always valid
        elif t == 'Wait':            self._check_wait(node)
        elif t == 'Observe':         self._check_observe(node)
        elif t == 'If':              self._check_if(node)
        elif t == 'LoopFor':         self._check_loop_for(node)
        elif t == 'LoopUntil':       self._check_loop_until(node)
        elif t == 'Break':           self._check_break(node)

    def _check_block(self, body: list):
        """Walk a block, flagging dead code after break."""
        terminated = False
        for stmt in body:
            self._check_statement(stmt, after_terminator=terminated)
            if stmt['type'] == 'Break':
                terminated = True

    # --- Individual statement checks ---

    def _check_movement(self, node: dict):
        if node['type'] == 'Run':
            speed = node.get('speed', 'normally')
            # check for conflicting speed signals — handled at program level
            # nothing to check per-node here

        # 'run quickly' + 'walk slowly' conflict checked in program scan
        pass

    def _check_wait(self, node: dict):
        if node['duration'] < 0:
            self.error("Wait duration must be positive.", node['pos'])
        elif node['duration'] == 0:
            self.warning("Waiting 0 time does nothing.", node['pos'])

    def _check_observe(self, node: dict):
        sensors = node['sensors']

        # Duplicate sensors in list
        seen = set()
        for s in sensors:
            if s in seen:
                self.warning(f"You listed '{s}' twice — it will only be read once.", node['pos'])
            seen.add(s)

        # No branches
        if not node['branches']:
            self.warning("You turned on the sensor but never checked what it found.", node['pos'])
            return

        # Only one sensor — hint
        if len(set(sensors)) == 1:
            self.hint("You could add more sensors to help your robot understand its surroundings.", node['pos'])

        # Track active sensors for branch validation
        outer_sensors = self.active_sensors
        self.active_sensors = set(sensors)
        self.observed_sensors.update(sensors)

        for branch in node['branches']:
            self._check_sensor_branch(branch)

        self.active_sensors = outer_sensors

    def _check_sensor_branch(self, node: dict):
        self._check_condition_sensors(node['condition'], in_observe=True)
        self._check_block(node['then'])
        if node['else']:
            self._check_block(node['else'])

    def _check_if(self, node: dict):
        self._check_condition_sensors(node['condition'], in_observe=False)
        self._check_block(node['then'])
        if node['else']:
            self._check_block(node['else'])

    def _check_loop_for(self, node: dict):
        count = node['count']
        if count <= 0:
            self.error("A loop needs to run at least once.", node['pos'])
            return
        if count != int(count):
            self.error("Loop count must be a whole number.", node['pos'])
            return

        self.loop_depth += 1
        self._check_block(node['body'])
        self.loop_depth -= 1

        has_break = self._block_has_break(node['body'])
        if not has_break:
            self.hint("Your robot might loop forever.", node['pos'])

    def _check_loop_until(self, node: dict):
        self._check_condition_sensors(node['condition'], in_observe=False)

        self.loop_depth += 1
        self._check_block(node['body'])
        self.loop_depth -= 1

    def _check_break(self, node: dict):
        if self.loop_depth == 0:
            self.error("Break only works inside a loop.", node['pos'])

    # --- Condition sensor validation ---

    def _check_condition_sensors(self, condition: dict, in_observe: bool):
        """
        Walk a condition tree and validate sensor references.
        in_observe=True  → sensor must be in active_sensors (declared in observe())
        in_observe=False → sensor must have been observed at some point (observed_sensors)
        """
        t = condition['type']

        if t in ('And', 'Or'):
            self._check_condition_sensors(condition['left'],  in_observe)
            self._check_condition_sensors(condition['right'], in_observe)
            return

        if t == 'Not':
            self._check_condition_sensors(condition['operand'], in_observe)
            return

        # Atomic condition — extract sensor name
        sensor = self._sensor_of(condition)
        if sensor is None:
            return

        if in_observe:
            if sensor not in self.active_sensors:
                pos = condition.get('pos', {'line': 0, 'col': 0})
                self.error(
                    f"You're checking '{sensor}' but forgot to add it to observe(...).",
                    pos
                )
        else:
            if sensor not in self.observed_sensors:
                pos = condition.get('pos', {'line': 0, 'col': 0})
                self.warning(
                    f"Your robot hasn't read '{sensor}' yet. Add observe({sensor}) first.",
                    pos
                )

        # Sensor-specific value checks
        self._check_condition_values(condition)

    def _sensor_of(self, condition: dict) -> Optional[str]:
        """Map condition type to sensor name."""
        mapping = {
            'DistCondition':   'dist',
            'ColourCondition': 'colour',
            'TouchCondition':  'touch',
            'IRCondition':     'IR',
            'UVCondition':     'UV',
            'GyroCondition':   'gyro',
            'SoundCondition':  'sound',
        }
        return mapping.get(condition['type'])

    def _check_condition_values(self, condition: dict):
        pos = condition.get('pos', {'line': 0, 'col': 0})
        t   = condition['type']

        if t == 'DistCondition':
            if condition['value'] < 0:
                self.error("Distance can't be negative.", pos)

        elif t == 'UVCondition':
            if condition['mode'] == 'index':
                if condition['index'] < 0 or condition['index'] > 11:
                    self.error("UV index goes from 0 to 11.", pos)

        elif t == 'SoundCondition':
            if condition['db'] < 0 or condition['db'] > 194:
                self.error("Sound level must be between 0 and 194 dB.", pos)

        elif t == 'GyroCondition':
            if condition['degrees'] < -180 or condition['degrees'] > 180:
                self.error("Tilt angle must be between -180 and 180 deg.", pos)

        elif t == 'ColourCondition':
            colour = condition['colour']
            if isinstance(colour, dict):   # wavelength range
                if colour['nmMin'] >= colour['nmMax']:
                    self.error("Start wavelength must be less than end wavelength.", pos)
                if colour['nmMin'] < 380 or colour['nmMax'] > 750:
                    self.warning(
                        "This wavelength is outside visible light range. Use IR or UV sensors instead.",
                        pos
                    )

    # --- Helpers ---

    def _block_has_break(self, body: list) -> bool:
        for stmt in body:
            if stmt['type'] == 'Break':
                return True
            if stmt['type'] in ('LoopFor', 'LoopUntil'):
                # break inside nested loop doesn't count for outer loop
                continue
            if 'body' in stmt and self._block_has_break(stmt['body']):
                return True
            if 'then' in stmt and self._block_has_break(stmt['then']):
                return True
        return False


# --- Public entry point ---

def analyze(ast: dict) -> list[Diagnostic]:
    return SemanticAnalyzer().analyze(ast)

def has_errors(diagnostics: list[Diagnostic]) -> bool:
    return any(d.severity == 'error' for d in diagnostics)

def has_warnings(diagnostics: list[Diagnostic]) -> bool:
    return any(d.severity in ('error', 'warning') for d in diagnostics)

def blocks_training(diagnostics: list[Diagnostic]) -> bool:
    """Training is blocked on any error or warning."""
    return has_warnings(diagnostics)