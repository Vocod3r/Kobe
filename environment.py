"""
Environment generator: compile Kobe IR into Gymnasium environments.

This is the core of Gate 2: generated environments must be semantically
equivalent to reference handwritten environments on deterministic test trajectories.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Callable, Optional


class IREnvironmentGenerator:
    """Converts Kobe IR into a Gymnasium environment specification."""
    
    def __init__(self, ir: list[dict]):
        self.ir = ir
        self.observation_spec = {}
        self.action_spec = {}
        self.objective_spec = {}
        self.termination_spec = {}
        self.safety_spec = {}
        self.reset_spec = {}
        self._analyze_ir()
    
    def _analyze_ir(self):
        """Walk the IR and extract environment semantics."""
        for instr in self.ir:
            op = instr['op']
            
            if op == 'POLICY':
                # Extract objective weights
                self.objective_spec = {
                    'curiosity': instr.get('curiosity', 0.3),
                    'safety': instr.get('safety', 0.5),
                    'comfort': instr.get('comfort', 0.5),
                    'efficiency': instr.get('efficiency', 0.5),
                }
            
            elif op == 'SENSE':
                # Record which sensors are observed
                for sensor in instr.get('sensors', []):
                    self.observation_spec[sensor] = self._sensor_spec(sensor)
            
            elif op in ('WALK', 'RUN'):
                # Record action bounds
                self.action_spec['move'] = {
                    'type': 'continuous',
                    'low': 0.0,
                    'high': 1.0,  # Aggressiveness in [0, 1]
                }
            
            elif op == 'TURN':
                # Turn is a discrete action
                if 'turn' not in self.action_spec:
                    self.action_spec['turn'] = {
                        'type': 'discrete',
                        'options': ['left', 'right'],
                    }
            
            elif op == 'STOP':
                # Stop is an action
                if 'stop' not in self.action_spec:
                    self.action_spec['stop'] = {'type': 'action'}
    
    def _sensor_spec(self, sensor: str) -> dict:
        """Define sensor observation specification."""
        specs = {
            'dist': {'type': 'continuous', 'low': 0.0, 'high': 200.0, 'unit': 'cm'},
            'colour': {'type': 'categorical', 'values': ['red', 'green', 'blue', 'none']},
            'touch': {'type': 'binary'},
            'IR': {'type': 'binary'},
            'UV': {'type': 'continuous', 'low': 0.0, 'high': 11.0},
            'gyro': {'type': 'continuous', 'low': -180.0, 'high': 180.0, 'unit': 'deg'},
            'sound': {'type': 'continuous', 'low': 0.0, 'high': 194.0, 'unit': 'db'},
        }
        return specs.get(sensor, {'type': 'unknown'})
    
    def generate_gymnasium_env(self, 
                               reset_fn: Callable,
                               step_fn: Callable) -> type:
        """Generate a Gymnasium environment class from IR.
        
        Args:
            reset_fn: Function that resets environment state; returns obs, info
            step_fn: Function that steps environment; takes action, returns obs, reward, done, truncated, info
        
        Returns:
            A Gymnasium Env subclass
        """
        ir = self.ir
        obs_spec = self.observation_spec
        action_spec = self.action_spec
        obj_spec = self.objective_spec
        
        class GeneratedRobotEnv(gym.Env):
            """Generated environment from Kobe IR."""
            
            def __init__(self):
                super().__init__()
                
                # Build observation space from sensor specifications
                obs_dict = {}
                for sensor, spec in obs_spec.items():
                    if spec['type'] == 'continuous':
                        obs_dict[sensor] = spaces.Box(
                            low=spec['low'], high=spec['high'],
                            shape=(1,), dtype=np.float32
                        )
                    elif spec['type'] == 'binary':
                        obs_dict[sensor] = spaces.Box(
                            low=0, high=1,
                            shape=(1,), dtype=np.int32
                        )
                    elif spec['type'] == 'categorical':
                        obs_dict[sensor] = spaces.Discrete(len(spec['values']))
                
                # Add priority weights to observation
                for key in obj_spec:
                    obs_dict[key] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                
                # Add progress indicator
                obs_dict['progress'] = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                
                self.observation_space = spaces.Dict(obs_dict)
                
                # Build action space: continuous aggressiveness in [0, 1]
                self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                
                # Store objectives and safety constraints
                self.objectives = obj_spec
                self.ir = ir
            
            def reset(self, seed=None, options=None):
                """Reset the environment."""
                super().reset(seed=seed)
                obs, info = reset_fn()
                return obs, info
            
            def step(self, action):
                """Step the environment."""
                obs, reward, done, truncated, info = step_fn(action)
                return obs, reward, done, truncated, info
            
            def render(self, mode='human'):
                """Render environment state."""
                pass
        
        return GeneratedRobotEnv
    
    def get_observation_spec(self) -> dict:
        """Return generated observation specification."""
        return self.observation_spec.copy()
    
    def get_action_spec(self) -> dict:
        """Return generated action specification."""
        return self.action_spec.copy()
    
    def get_objective_spec(self) -> dict:
        """Return generated objective specification."""
        return self.objective_spec.copy()
    
    def get_ir_summary(self) -> dict:
        """Summary of what was generated from the IR."""
        return {
            'observations': self.observation_spec,
            'actions': self.action_spec,
            'objectives': self.objective_spec,
            'termination': self.termination_spec,
            'safety': self.safety_spec,
            'ir_length': len(self.ir),
            'ir_operations': list(set(instr['op'] for instr in self.ir)),
        }


def create_kobe_environment(ir: list[dict],
                            interpreter_class: type,
                            priorities: dict) -> gym.Env:
    """
    High-level factory: create a Gymnasium environment from Kobe IR.
    
    Uses the reference interpreter as ground truth for semantics.
    
    Args:
        ir: Compiled Kobe IR
        interpreter_class: Reference interpreter class (e.g., IRInterpreter from backend.py)
        priorities: Policy priorities dict
    
    Returns:
        A configured Gymnasium environment instance
    """
    interpreter = interpreter_class(ir, priorities)
    generator = IREnvironmentGenerator(ir)
    
    def reset_fn():
        interpreter.reset()
        return interpreter.get_obs(), {}
    
    def step_fn(action):
        a = float(action[0]) if hasattr(action, '__len__') else float(action)
        result = interpreter.step(a)
        obs = interpreter.get_obs()
        reward = interpreter.compute_reward()
        done = result['done']
        return obs, reward, done, False, {}
    
    env_class = generator.generate_gymnasium_env(reset_fn, step_fn)
    return env_class()
