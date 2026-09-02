"""
IR trace executor: walks compiled IR against the same scenario, produces equivalent traces.

This independently executes the compiled IR (jump-threaded) against the same
scripted sensor scenario as the reference interpreter, producing traces for comparison.
"""

from dataclasses import dataclass
import numpy as np
from reference_interpreter import Scenario, Trace, TraceEvent


class IRTraceExecutor:
    """Executes compiled IR against a Scenario, producing a trace."""
    
    def __init__(self, ir: list[dict], scenario: Scenario, max_steps: int | None = None):
        self.ir = ir
        self.scenario = scenario
        self.trace = Trace()
        self.step = 0
        self.pc = 0
        self.max_steps = max_steps
        self.hit_cap = False
        self.loop_stack = []  # Stack of (pc_to_return_to, iteration_count)
        self.condition_stack = []
        self.observed_sensors = {}
        self.halted = False
    
    def execute(self) -> Trace:
        """Execute the IR and return the trace."""
        while self.pc < len(self.ir) and not self.halted:
            if self.max_steps is not None and self.step > self.max_steps:
                self.hit_cap = True
                break
            self._step_instruction()
        
        if not self.halted:
            self.trace.add_halt(self.step)
        
        return self.trace
    
    def _step_instruction(self):
        """Execute one IR instruction."""
        instr = self.ir[self.pc]
        op = instr['op']
        
        if op in ('ALGORITHM', 'HARDWARE', 'POLICY'):
            self.pc += 1
        
        elif op == 'HALT':
            self.trace.add_halt(self.step)
            self.halted = True
        
        elif op == 'WALK':
            direction = instr.get('direction', 'forward')
            self.trace.add_action(self.step, 'walk', direction=direction, speedMultiplier=instr.get('speedMultiplier', 1.0))
            self._advance_sensors()
            self.step += 1
            self.pc += 1
        
        elif op == 'RUN':
            direction = instr.get('direction', 'forward')
            self.trace.add_action(self.step, 'run', direction=direction, speedMultiplier=instr.get('speedMultiplier', 2.0))
            self._advance_sensors()
            self.step += 1
            self.pc += 1
        
        elif op == 'STOP':
            self.trace.add_action(self.step, 'stop')
            self.step += 1
            self.pc += 1
        
        elif op == 'TURN':
            direction = instr.get('direction', 'left')
            self.trace.add_action(self.step, 'turn', direction=direction)
            self.step += 1
            self.pc += 1
        
        elif op == 'WAIT':
            duration = instr.get('durationMs', instr.get('duration', 1))
            self.trace.add_action(self.step, 'wait', duration=duration)
            self.step += 1
            self.pc += 1
        
        elif op == 'SENSE':
            sensors = instr.get('sensors', [])
            readings = {}
            for sensor in sensors:
                readings[sensor] = self.scenario.get_reading(sensor, self.step)
                self.observed_sensors[sensor] = readings[sensor]
            self.trace.add_observe(self.step, sensors, readings)
            self.step += 1
            self.pc += 1
        
        elif op == 'CMP_DIST':
            dist = self.observed_sensors.get('dist', self.scenario.distance_readings[0])
            result = self._compare(dist, instr['comparator'], instr['valueCm'])
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_COLOUR':
            colour = self.observed_sensors.get('colour', {'name': 'none', 'nm': 600})
            if isinstance(colour, dict):
                colour_name = colour.get('name', 'none')
            else:
                colour_name = colour
            result = (colour_name == instr['colour']) if isinstance(instr['colour'], str) else True
            result = (not result) if instr.get('negate', False) else result
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_TOUCH':
            touch = self.observed_sensors.get('touch', False)
            result = bool(touch)
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_IR':
            ir_reading = self.observed_sensors.get('IR', False)
            if instr.get('mode') == 'detected':
                result = bool(ir_reading)
            else:
                result = self._compare(ir_reading, instr.get('comparator', '=='), instr.get('value', 0))
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_UV':
            uv_reading = self.observed_sensors.get('UV', 0.0)
            if instr.get('mode') == 'detected':
                result = uv_reading > 0
            else:
                result = self._compare(uv_reading, instr.get('comparator', '=='), instr.get('index', 0))
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_GYRO':
            gyro_reading = self.observed_sensors.get('gyro', 0.0)
            result = self._compare(gyro_reading, instr['comparator'], instr['degrees'])
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'CMP_SOUND':
            sound_reading = self.observed_sensors.get('sound', 30.0)
            result = self._compare(sound_reading, instr['comparator'], instr['db'])
            self.condition_stack.append(result)
            self.pc += 1
        
        elif op == 'AND':
            b = self.condition_stack.pop()
            a = self.condition_stack.pop()
            self.condition_stack.append(a and b)
            self.pc += 1
        
        elif op == 'OR':
            b = self.condition_stack.pop()
            a = self.condition_stack.pop()
            self.condition_stack.append(a or b)
            self.pc += 1
        
        elif op == 'NOT':
            a = self.condition_stack.pop()
            self.condition_stack.append(not a)
            self.pc += 1
        
        elif op == 'JUMP_IF_FALSE':
            condition_result = self.condition_stack.pop() if self.condition_stack else False
            
            # Emit branch event only if this is explicitly marked as a loop check
            if instr.get('isLoopCheck', False):
                self.trace.add_branch(self.step, condition_result, 
                                    condition_str=f"until ...", 
                                    isLoopCheck=True)
                self.step += 1
            else:
                # For regular if/else branches, emit the branch event
                self.trace.add_branch(self.step, condition_result, 
                                    condition_str=instr.get('condition_str', '?'))
                self.step += 1
            
            self.pc = (self.pc + 1) if condition_result else instr['target']
        
        elif op == 'JUMP_IF_TRUE':
            condition_result = self.condition_stack.pop() if self.condition_stack else False

            # Emit branch event for loop checks (loop_until exit check)
            if instr.get('isLoopCheck', True):
                self.trace.add_branch(self.step, condition_result,
                                     condition_str=instr.get('condition_str', '?'),
                                     isLoopCheck=True)
            else:
                self.trace.add_branch(self.step, condition_result,
                                     condition_str=instr.get('condition_str', '?'))
            self.step += 1

            # Jump to exit when condition is True (loop_until exit semantics)
            self.pc = instr['target'] if condition_result else (self.pc + 1)

        elif op == 'JUMP':
            self.pc = instr['target']
        
        elif op == 'LOOP_START':
            count = instr['count']
            self.loop_stack.append([count, self.pc])
            self.pc += 1
        
        elif op == 'LOOP_END':
            if self.loop_stack:
                self.loop_stack[-1][0] -= 1
                if self.loop_stack[-1][0] > 0:
                    self.pc = self.loop_stack[-1][1] + 1
                else:
                    self.loop_stack.pop()
                    self.pc += 1
            else:
                self.pc += 1
        
        elif op == 'BREAK':
            if self.loop_stack:
                self.loop_stack.pop()
            self.pc = instr['target']
        
        else:
            self.pc += 1
    
    def _compare(self, a, op: str, b) -> bool:
        """Compare two values with an operator."""
        return {
            '<': a < b,
            '<=': a <= b,
            '>': a > b,
            '>=': a >= b,
            '==': a == b,
            '!=': a != b,
        }.get(op, False)
    
    def _advance_sensors(self):
        """Advance sensor state after an action (simulate distance change, etc.)."""
        # In a real scenario with movement detection, we'd update distance here
        pass
