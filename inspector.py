"""
Inspector: show generated environment/training semantics to users.

This is critical for the research question about transparency.
Users must be able to see what Kobe generated from their specification.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from environment import IREnvironmentGenerator
from trainer import TrainerGenerator


class IRInspector:
    """Inspects and explains Kobe IR and generated artifacts."""
    
    def __init__(self, ir: list[dict], priorities: dict):
        self.ir = ir
        self.priorities = priorities
        self.env_gen = IREnvironmentGenerator(ir)
        self.trainer_gen = TrainerGenerator(ir, priorities)
    
    def explain_observations(self) -> dict:
        """Explain what observations the robot will make."""
        spec = self.env_gen.get_observation_spec()
        
        explanations = {}
        for sensor, sensor_spec in spec.items():
            explanations[sensor] = {
                'type': sensor_spec['type'],
                'description': self._sensor_description(sensor),
                'range': self._sensor_range(sensor_spec),
                'units': sensor_spec.get('unit', 'none'),
            }
        
        return {
            'summary': f'Robot observes {len(spec)} sensors',
            'sensors': explanations,
        }
    
    def explain_actions(self) -> dict:
        """Explain what actions the robot can take."""
        spec = self.env_gen.get_action_spec()
        
        explanations = {}
        for action, action_spec in spec.items():
            explanations[action] = {
                'type': action_spec['type'],
                'description': self._action_description(action),
                'bounds': action_spec.get('bounds', 'discrete'),
            }
        
        return {
            'summary': f'Robot can perform {len(spec)} action types',
            'actions': explanations,
        }
    
    def explain_objectives(self) -> dict:
        """Explain what Kobe is optimizing for."""
        obj = self.env_gen.get_objective_spec()
        
        explanations = {}
        for key, weight in obj.items():
            explanations[key] = {
                'weight': weight,
                'description': self._objective_description(key),
                'interpretation': self._objective_interpretation(key, weight),
            }
        
        return {
            'summary': 'Kobe will optimize for these goals (weighted combination)',
            'objectives': explanations,
            'note': 'Weights are your policy slider values; higher = more important',
        }
    
    def explain_termination(self) -> dict:
        """Explain when an episode ends."""
        return {
            'summary': 'Episode ends when:',
            'conditions': [
                'Robot reaches the goal',
                'Time limit exceeded (max steps)',
                'Robot violates a safety constraint',
                'Task explicitly fails',
            ],
            'note': 'Termination is deterministic; same scenario always ends the same way',
        }
    
    def explain_safety(self) -> dict:
        """Explain safety constraints."""
        safety_weight = self.priorities.get('safety', 0.5)
        
        return {
            'summary': 'Safety constraints prevent dangerous behavior',
            'priority': safety_weight,
            'constraints': [
                'Robot cannot collide',
                'Robot cannot exceed safe speed',
                'Robot must respect action bounds',
            ],
            'enforcement': 'Hard constraint (cannot be violated even for reward)',
            'note': f'Safety weight {safety_weight} means this is {"very" if safety_weight > 0.7 else "moderately" if safety_weight > 0.3 else "less"} important',
        }
    
    def explain_training(self) -> dict:
        """Explain how Kobe will train the policy."""
        spec = self.trainer_gen.get_training_spec()
        params = spec['hyperparameters']
        
        return {
            'summary': f'Kobe uses {spec["algorithm"]} algorithm to train the robot',
            'algorithm': spec['algorithm'],
            'learning_rate': params['learning_rate'],
            'num_training_steps': params['num_training_steps'],
            'objectives': spec['objectives'],
            'key_hyperparameters': {
                'discount_factor': params['discount_factor'],
                'batch_size': params['batch_size'],
                'replay_buffer_size': params['replay_buffer_size'],
            },
            'explanation': f'''
The robot will:
1. Collect experience by taking random/learned actions
2. Learn from replay buffer of past experiences
3. Update its policy to maximize the weighted objectives
4. Repeat {params['num_training_steps']} times

Training takes ~{params['num_training_steps'] // 1000}k environment steps, typically 5-30 minutes.
            ''',
        }
    
    def full_inspection_report(self) -> dict:
        """Complete inspection report for user debugging."""
        return {
            'ir_summary': self.env_gen.get_ir_summary(),
            'observations': self.explain_observations(),
            'actions': self.explain_actions(),
            'objectives': self.explain_objectives(),
            'termination': self.explain_termination(),
            'safety': self.explain_safety(),
            'training': self.explain_training(),
            'generated_ir_length': len(self.ir),
            'generated_ir_operations': list(set(instr['op'] for instr in self.ir)),
        }
    
    # ── Internal explanation helpers ──
    
    def _sensor_description(self, sensor: str) -> str:
        descriptions = {
            'dist': 'Distance to nearest obstacle (ultrasonic sensor)',
            'colour': 'Detected colour of surface below robot',
            'touch': 'Touch sensor pressed or released',
            'IR': 'Infrared sensor detects objects',
            'UV': 'UV light intensity',
            'gyro': 'Robot tilt angle',
            'sound': 'Ambient sound level',
        }
        return descriptions.get(sensor, f'Sensor: {sensor}')
    
    def _sensor_range(self, spec: dict) -> str:
        if spec['type'] == 'continuous':
            return f"{spec['low']} to {spec['high']}"
        elif spec['type'] == 'categorical':
            return f"One of: {spec['values']}"
        elif spec['type'] == 'binary':
            return 'True or False'
        return 'Unknown range'
    
    def _action_description(self, action: str) -> str:
        descriptions = {
            'move': 'Move forward (aggressiveness 0-1)',
            'turn': 'Turn left or right',
            'stop': 'Stop moving',
        }
        return descriptions.get(action, f'Action: {action}')
    
    def _objective_description(self, objective: str) -> str:
        descriptions = {
            'efficiency': 'Complete task quickly with minimal energy',
            'safety': 'Avoid collisions and dangerous states',
            'comfort': 'Move smoothly without sudden acceleration',
            'curiosity': 'Explore diverse states',
        }
        return descriptions.get(objective, f'Objective: {objective}')
    
    def _objective_interpretation(self, objective: str, weight: float) -> str:
        if weight < 0.2:
            priority = 'Not important'
        elif weight < 0.4:
            priority = 'Low priority'
        elif weight < 0.6:
            priority = 'Medium priority'
        elif weight < 0.8:
            priority = 'High priority'
        else:
            priority = 'Very high priority'
        
        return f'{priority} (weight = {weight:.2f})'


def inspect_ir(ir: list[dict], priorities: dict, 
               output_path: Optional[str] = None) -> dict:
    """
    Inspect compiled IR and return human-readable explanation.
    
    Args:
        ir: Compiled Kobe IR
        priorities: Policy priorities
        output_path: Optional file to write inspection report as JSON
    
    Returns:
        Inspection report dict
    """
    inspector = IRInspector(ir, priorities)
    report = inspector.full_inspection_report()
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


def print_inspection_report(ir: list[dict], priorities: dict):
    """Pretty-print inspection report for terminal viewing."""
    inspector = IRInspector(ir, priorities)
    report = inspector.full_inspection_report()
    
    print("\n" + "="*70)
    print("KOBE INSPECTION REPORT")
    print("="*70)
    
    print("\n[OBSERVATIONS]")
    for sensor, info in report['observations']['sensors'].items():
        print(f"  {sensor}: {info['description']}")
        print(f"    Range: {info['range']} {info['units']}")
    
    print("\n[ACTIONS]")
    for action, info in report['actions']['actions'].items():
        print(f"  {action}: {info['description']}")
    
    print("\n[OBJECTIVES]")
    for obj, info in report['objectives']['objectives'].items():
        print(f"  {obj}: {info['interpretation']}")
        print(f"    {info['description']}")
    
    print("\n[SAFETY]")
    print(f"  Safety priority: {report['safety']['priority']}")
    for constraint in report['safety']['constraints']:
        print(f"    - {constraint}")
    
    print("\n[TRAINING]")
    print(f"  Algorithm: {report['training']['algorithm']}")
    print(f"  Steps: {report['training']['num_training_steps']:,}")
    print(f"  Learning rate: {report['training']['learning_rate']}")
    
    print("\n" + "="*70)
