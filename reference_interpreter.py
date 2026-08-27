"""
Reference interpreter: walks the AST directly, produces deterministic traces.

This is independent of the compiler and executes source programs by tree-recursion,
not flattened jumps. It takes a scripted Scenario of sensor readings and produces
a Trace showing every action/observation/branch decision.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


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
    
    def __init__(self, ast: dict, scenario: Scenario):
        self.ast = ast
        self.scenario = scenario
        self.trace = Trace()
        self.step = 0
        self.observed_sensors = {}
        self.halted = False
        self.break_exception = None
    
    def execute(self) -> Trace:
        """Execute the entire program and return the trace."""
        try:
            for stmt in self.ast['body']:
                if self.halted:
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
            sensors = stmt.get('sensors', [])
            readings = {}
            for sensor in sensors:
                readings[sensor] = self.scenario.get_reading(sensor, self.step)
                self.observed_sensors[sensor] = readings[sensor]
            self.trace.add_observe(self.step, sensors, readings)
            self.step += 1
        
        elif stmt_type == 'If':
            self._execute_if(stmt)
        
        elif stmt_type == 'LoopFor':
            self._execute_loop_for(stmt)
        
        elif stmt_type == 'LoopUntil':
            self._execute_loop_until(stmt)
        
        elif stmt_type == 'Break':
            raise BreakException()
    
    def _execute_if(self, node: dict):
        """Execute an if statement with possibly multiple branches."""
        condition_value = self._eval_condition(node['condition'])
        self.trace.add_branch(self.step, condition_value, condition_str=self._condition_str(node['condition']))
        self.step += 1
        
        if condition_value:
            for stmt in node['then_body']:
                if self.halted:
                    return
                self._execute_statement(stmt)
        else:
            for else_if in node.get('else_if', []):
                condition_value = self._eval_condition(else_if['condition'])
                self.trace.add_branch(self.step, condition_value, condition_str=self._condition_str(else_if['condition']))
                self.step += 1
                
                if condition_value:
                    for stmt in else_if['body']:
                        if self.halted:
                            return
                        self._execute_statement(stmt)
                    return
            
            # else clause
            for stmt in node.get('else_body', []):
                if self.halted:
                    return
                self._execute_statement(stmt)
    
    def _execute_loop_for(self, node: dict):
        """Execute a loop for N times."""
        count = node.get('count', 1)
        body = node.get('body', [])
        
        for iteration in range(count):
            if self.halted:
                break
            self.trace.add_loop_iter(self.step, iteration=iteration, count=count)
            
            try:
                for stmt in body:
                    if self.halted:
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
        """Evaluate a condition expression recursively."""
        cond_type = cond['type']
        
        if cond_type == 'Comparison':
            sensor = cond['sensor']
            comparator = cond['comparator']
            value = cond['value']
            
            # Get the sensor reading (from observed_sensors if already read, else from scenario)
            if sensor in self.observed_sensors:
                reading = self.observed_sensors[sensor]
            else:
                reading = self.scenario.get_reading(sensor, self.step)
            
            if isinstance(reading, dict):  # colour sensor
                if comparator == '==':
                    return reading.get('name') == value
                elif comparator == '!=':
                    return reading.get('name') != value
                return False
            
            # Numeric comparison
            return {
                '<': reading < value,
                '<=': reading <= value,
                '>': reading > value,
                '>=': reading >= value,
                '==': reading == value,
                '!=': reading != value,
            }.get(comparator, False)
        
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
        
        if cond_type == 'Comparison':
            return f"{cond['sensor']} {cond['comparator']} {cond['value']}"
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
