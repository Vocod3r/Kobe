"""
Reference interpreter: walks the AST directly, produces deterministic traces.

This is independent of the compiler and executes source programs by tree-recursion,
not flattened jumps. It takes a scripted Scenario of sensor readings and produces
a Trace showing every action/observation/branch decision.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from compiler import _to_cm


@dataclass
class Scenario:
    """Scripted sensor readings for deterministic execution."""
    distance_readings: list[float] = field(default_factory=lambda: [100.0])
    colour_readings: list[dict] = field(default_factory=lambda: [{'name': 'none', 'nm': 600}])
    ir_readings: list[bool] = field(default_factory=lambda: [False])
    touch_readings: list[bool] = field(default_factory=lambda: [False])
    uv_readings: list[float] = field(default_factory=lambda: [0.0])
    gyro_readings: list[float] = field(default_factory=lambda: [0.0])
    sound_readings: list[float] = field(default_factory=lambda: [30.0])
    
    # Movement happens; simulate sensor change per step
    distance_change_per_step: float = 0.0  # How much distance changes after each action
    
    def get_reading(self, sensor: str, step: int) -> Any:
        """Get a sensor reading at a specific step, cycling if needed."""
        readings_map = {
            'dist': self.distance_readings,
            'colour': self.colour_readings,
            'IR': self.ir_readings,
            'touch': self.touch_readings,
            'UV': self.uv_readings,
            'gyro': self.gyro_readings,
            'sound': self.sound_readings,
        }
        readings = readings_map.get(sensor, [])
        if not readings:
            return None
        return readings[step % len(readings)]


@dataclass
class TraceEvent:
    """A single event in the execution trace."""
    kind: str  # 'action', 'observe', 'branch', 'break', 'halt', 'loop_iter'
    step: int
    details: dict = field(default_factory=dict)


@dataclass
class Trace:
    """Complete execution trace: sequence of events."""
    events: list[TraceEvent] = field(default_factory=list)
    
    def add_action(self, step: int, action: str, **kwargs):
        self.events.append(TraceEvent('action', step, {'action': action, **kwargs}))
    
    def add_observe(self, step: int, sensors: list[str], readings: dict, **kwargs):
        self.events.append(TraceEvent('observe', step, {'sensors': sensors, 'readings': readings, **kwargs}))
    
    def add_branch(self, step: int, condition: bool, condition_str: str = '', **kwargs):
        self.events.append(TraceEvent('branch', step, {'result': condition, 'condition': condition_str, **kwargs}))
    
    def add_break(self, step: int, **kwargs):
        self.events.append(TraceEvent('break', step, kwargs))
    
    def add_halt(self, step: int, **kwargs):
        self.events.append(TraceEvent('halt', step, kwargs))
    
    def add_loop_iter(self, step: int, **kwargs):
        """Informational only; not compared in equivalence."""
        self.events.append(TraceEvent('loop_iter', step, kwargs))
    
    def observable_events(self) -> list[TraceEvent]:
        """Return only observable events (not loop_iter)."""
        return [e for e in self.events if e.kind != 'loop_iter']


class ReferenceInterpreter:
    """Interprets Kobe AST directly, producing deterministic traces."""
    
    def __init__(self, ast: dict, scenario: Scenario, max_steps: int | None = None):
        self.ast = ast
        self.scenario = scenario
        self.trace = Trace()
        self.step = 0
        self.max_steps = max_steps
        self.hit_cap = False
        self.observed_sensors = {}
        self.halted = False
        self.break_exception = None
    
    def execute(self) -> Trace:
        """Execute the entire program and return the trace."""
        try:
            for stmt in self.ast['body']:
                if self.halted or self.hit_cap:
                    break
                self._execute_statement(stmt)
        except BreakException:
            # Break outside a loop context (shouldn't happen in well-formed programs)
            pass
        
        if not self.halted:
            self.trace.add_halt(self.step)
        
        return self.trace
    
    def _execute_statement(self, stmt: dict):
        """Execute a single statement."""
        if self.halted:
            return
        if self.max_steps is not None and self.step > self.max_steps:
            self.hit_cap = True
            return
        
        stmt_type = stmt['type']
        
        if stmt_type == 'Walk':
            direction = stmt.get('direction', 'forward')
            speed = stmt.get('speed', 'normally')
            self.trace.add_action(self.step, 'walk', direction=direction, speed=speed)
            self._advance_sensors()
            self.step += 1
        
        elif stmt_type == 'Run':
            direction = stmt.get('direction', 'forward')
            speed = stmt.get('speed', 'normally')
            self.trace.add_action(self.step, 'run', direction=direction, speed=speed)
            self._advance_sensors()
            self.step += 1
        
        elif stmt_type == 'Turn':
            direction = stmt.get('direction', 'left')
            self.trace.add_action(self.step, 'turn', direction=direction)
            self.step += 1
        
        elif stmt_type == 'Stop':
            self.trace.add_action(self.step, 'stop')
            self.step += 1
        
        elif stmt_type == 'Wait':
            duration = stmt.get('duration', 1)
            unit = stmt.get('unit', 'sec')
            self.trace.add_action(self.step, 'wait', duration=duration, unit=unit)
            self.step += 1
        
        elif stmt_type == 'Observe':
            self._execute_observe(stmt)
        
        elif stmt_type == 'If':
            self._execute_if(stmt)
        
        elif stmt_type == 'LoopFor':
            self._execute_loop_for(stmt)
        
        elif stmt_type == 'LoopUntil':
            self._execute_loop_until(stmt)
        
        elif stmt_type == 'Break':
            raise BreakException()
    
    def _execute_observe(self, node: dict):
        """Execute an observe block: emit an observe event, then evaluate each
        sensor branch independently (branches are NOT an if/elif chain — every
        branch's condition is evaluated regardless of prior branch results,
        matching compiler.compile_observe)."""
        sensors = node.get('sensors', [])
        readings = {}
        for sensor in sensors:
            readings[sensor] = self.scenario.get_reading(sensor, self.step)
            self.observed_sensors[sensor] = readings[sensor]
        self.trace.add_observe(self.step, sensors, readings)
        self.step += 1

        for branch in node.get('branches', []):
            if self.halted or (self.max_steps is not None and self.step > self.max_steps):
                self.hit_cap = True
                return
            condition_value = self._eval_condition(branch['condition'])
            self.trace.add_branch(self.step, condition_value, condition_str=self._condition_str(branch['condition']))
            self.step += 1

            if condition_value:
                for stmt in branch['then']:
                    if self.halted:
                        return
                    self._execute_statement(stmt)
            else:
                for stmt in branch.get('else', []):
                    if self.halted:
                        return
                    self._execute_statement(stmt)

    def _execute_if(self, node: dict):
        """Execute a standalone if/then/else statement."""
        condition_value = self._eval_condition(node['condition'])
        self.trace.add_branch(self.step, condition_value, condition_str=self._condition_str(node['condition']))
        self.step += 1

        if condition_value:
            for stmt in node['then']:
                if self.halted:
                    return
                self._execute_statement(stmt)
        else:
            for stmt in node.get('else', []):
                if self.halted:
                    return
                self._execute_statement(stmt)
    
    def _execute_loop_for(self, node: dict):
        """Execute a loop for N times."""
        count = int(node.get('count', 1))
        body = node.get('body', [])
        
        for iteration in range(count):
            if self.halted:
                break
            if self.max_steps is not None and self.step > self.max_steps:
                self.hit_cap = True
                break
            self.trace.add_loop_iter(self.step, iteration=iteration, count=count)
            
            try:
                for stmt in body:
                    if self.halted or self.hit_cap:
                        break
                    self._execute_statement(stmt)
            except BreakException:
                break
    
    def _execute_loop_until(self, node: dict):
        """Execute a loop until condition becomes true."""
        condition = node['condition']
        body = node.get('body', [])
        iteration = 0
        
        while True:
            if self.halted:
                break
            if self.max_steps is not None and self.step > self.max_steps:
                self.hit_cap = True
                break
            
            # Evaluate the condition (note: loop_until means "until condition is true")
            condition_value = self._eval_condition(condition)
            self.trace.add_branch(self.step, condition_value, 
                                condition_str=f"until {self._condition_str(condition)}", 
                                isLoopCheck=True)
            self.step += 1
            
            # Exit when condition becomes true
            if condition_value:
                break
            
            self.trace.add_loop_iter(self.step, iteration=iteration)
            
            try:
                for stmt in body:
                    if self.halted:
                        break
                    self._execute_statement(stmt)
            except BreakException:
                break
            
            iteration += 1
    
    def _eval_condition(self, cond: dict) -> bool:
        """Evaluate a condition expression recursively.

        Mirrors ir_trace_executor.IRTraceExecutor's CMP_* handling exactly —
        including its fallback defaults when a sensor hasn't been observed yet —
        so the two executors are semantically equivalent."""
        cond_type = cond['type']

        if cond_type == 'DistCondition':
            dist = self.observed_sensors.get('dist', self.scenario.distance_readings[0])
            value_cm = _to_cm(cond['value'], cond['unit'])
            return _compare(dist, cond['comparator'], value_cm)

        elif cond_type == 'ColourCondition':
            colour = self.observed_sensors.get('colour', {'name': 'none', 'nm': 600})
            colour_name = colour.get('name', 'none') if isinstance(colour, dict) else colour
            result = (colour_name == cond['colour']) if isinstance(cond['colour'], str) else True
            return (not result) if cond['negate'] else result

        elif cond_type == 'TouchCondition':
            return bool(self.observed_sensors.get('touch', False))

        elif cond_type == 'IRCondition':
            ir_reading = self.observed_sensors.get('IR', False)
            if cond['mode'] == 'detected':
                return bool(ir_reading)
            return _compare(ir_reading, cond['comparator'], cond['value'])

        elif cond_type == 'UVCondition':
            uv_reading = self.observed_sensors.get('UV', 0.0)
            if cond['mode'] == 'detected':
                return uv_reading > 0
            return _compare(uv_reading, cond['comparator'], cond['index'])

        elif cond_type == 'GyroCondition':
            gyro_reading = self.observed_sensors.get('gyro', 0.0)
            return _compare(gyro_reading, cond['comparator'], cond['degrees'])

        elif cond_type == 'SoundCondition':
            sound_reading = self.observed_sensors.get('sound', 30.0)
            return _compare(sound_reading, cond['comparator'], cond['db'])

        elif cond_type == 'And':
            left = self._eval_condition(cond['left'])
            right = self._eval_condition(cond['right'])
            return left and right

        elif cond_type == 'Or':
            left = self._eval_condition(cond['left'])
            right = self._eval_condition(cond['right'])
            return left or right

        elif cond_type == 'Not':
            operand = self._eval_condition(cond['operand'])
            return not operand

        return False

    def _condition_str(self, cond: dict) -> str:
        """Generate a human-readable condition string for tracing."""
        cond_type = cond['type']

        if cond_type == 'DistCondition':
            return f"dist {cond['comparator']} {cond['value']}{cond['unit']}"
        elif cond_type == 'ColourCondition':
            neg = 'not ' if cond['negate'] else ''
            return f"colour is {neg}{cond['colour']}"
        elif cond_type == 'TouchCondition':
            return "touch pressed"
        elif cond_type == 'IRCondition':
            if cond['mode'] == 'detected':
                return "IR detected"
            return f"IR signal {cond['comparator']} {cond['value']}"
        elif cond_type == 'UVCondition':
            if cond['mode'] == 'detected':
                return "UV detected"
            return f"UV index {cond['comparator']} {cond['index']}"
        elif cond_type == 'GyroCondition':
            return f"gyro tilt {cond['comparator']} {cond['degrees']}deg"
        elif cond_type == 'SoundCondition':
            return f"sound {cond['comparator']} {cond['db']}db"
        elif cond_type == 'And':
            return f"({self._condition_str(cond['left'])} and {self._condition_str(cond['right'])})"
        elif cond_type == 'Or':
            return f"({self._condition_str(cond['left'])} or {self._condition_str(cond['right'])})"
        elif cond_type == 'Not':
            return f"not ({self._condition_str(cond['operand'])})"
        return "?"
    
    def _advance_sensors(self):
        """Advance sensor state after a step (simulate distance change, etc.)."""
        pass


class BreakException(Exception):
    """Used to implement break statement by exception."""
    pass


def _compare(a, op: str, b) -> bool:
    return {
        '<': a < b, '<=': a <= b, '>': a > b,
        '>=': a >= b, '==': a == b, '!=': a != b,
    }.get(op, False)