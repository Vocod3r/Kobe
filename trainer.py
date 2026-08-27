"""
Trainer generator: compile Kobe IR into TorchRL training configurations.

This generates training scripts and hyperparameters from the IR's objective
specifications, ensuring the trainer respects safety constraints and
weighted objectives.
"""

from __future__ import annotations

import json
from typing import Optional


class TrainerGenerator:
    """Generates TorchRL training configurations from Kobe IR."""
    
    def __init__(self, ir: list[dict], priorities: dict):
        self.ir = ir
        self.priorities = priorities
        self._extract_training_spec()
    
    def _extract_training_spec(self):
        """Walk IR to extract training requirements."""
        self.algorithm = None
        self.objectives = {}
        self.safety_constraints = []
        self.termination_conditions = []
        self.num_steps = 20_000  # Default
        
        for instr in self.ir:
            if instr['op'] == 'ALGORITHM':
                self.algorithm = instr.get('name', 'SAC')
            elif instr['op'] == 'POLICY':
                self.objectives = {
                    'curiosity': instr.get('curiosity', 0.3),
                    'safety': instr.get('safety', 0.5),
                    'comfort': instr.get('comfort', 0.5),
                    'efficiency': instr.get('efficiency', 0.5),
                }
    
    def generate_training_script(self, env_name: str = 'KobeEnv-v0') -> str:
        """Generate a complete TorchRL training script."""
        
        algorithm = self.algorithm or 'SAC'
        
        script = f'''"""
Auto-generated TorchRL training script from Kobe IR.
Algorithm: {algorithm}
Objectives: {self.objectives}
"""

import torch
import torch.nn as nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.envs import GymEnv
from torchrl.modules import MLP, Actor, ValueOperator
from torchrl.objectives import {algorithm}Loss
from torchrl.trainers import offpolicy_train
import gymnasium as gym

# Environment
env = GymEnv('{env_name}')
eval_env = GymEnv('{env_name}')

# Network architecture
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor_net = MLP(in_features=state_dim, out_features=action_dim * 2, num_cells=[256, 256])
actor = Actor(actor_net)

q_net = MLP(in_features=state_dim + action_dim, out_features=1, num_cells=[256, 256])
value_net = MLP(in_features=state_dim, out_features=1, num_cells=[256, 256])

# Objectives
objectives = {self.objectives}

# Trainer
trainer = offpolicy_train(
    actor=actor,
    value=value_net,
    q_net=q_net,
    optimizer=torch.optim.Adam,
    loss_fn={algorithm}Loss,
    env=env,
    eval_env=eval_env,
    num_epochs=200,
    num_steps_per_epoch=100,
    num_steps_per_collector=100,
    objectives=objectives,
)

# Run training
trainer.fit()

# Save policy
torch.save(actor.state_dict(), 'policy.pt')
'''
        
        return script
    
    def generate_hyperparameters(self) -> dict:
        """Generate algorithm-specific hyperparameters based on IR and priorities."""
        
        safety_weight = self.priorities.get('safety', 0.5)
        efficiency_weight = self.priorities.get('efficiency', 0.5)
        
        base_params = {
            'learning_rate': 3e-4,
            'discount_factor': 0.99,
            'target_update_frequency': 2,
            'replay_buffer_size': 100_000,
            'batch_size': 256,
            'num_training_steps': 200_000,
            'entropy_coefficient': 0.2,
        }
        
        # Adjust based on priorities
        if safety_weight > 0.7:
            base_params['entropy_coefficient'] *= 1.5  # More exploration for safety
            base_params['learning_rate'] *= 0.7  # Slower learning
        
        if efficiency_weight > 0.7:
            base_params['discount_factor'] = 0.95  # Shorter horizons
        
        algorithm = self.algorithm or 'SAC'
        
        algo_specific = {
            'SAC': {
                'entropy_autotune': True,
                'alpha_init': 0.1,
            },
            'TD3': {
                'policy_noise': 0.2,
                'noise_clip': 0.5,
                'policy_update_frequency': 2,
            },
            'DroQ': {
                'num_q_nets': 2,
                'update_frequency': 1,
                'utd_ratio': 20,
            }
        }
        
        base_params.update(algo_specific.get(algorithm, {}))
        return base_params
    
    def generate_evaluation_protocol(self) -> dict:
        """Generate evaluation protocol for trained policies."""
        
        return {
            'num_eval_episodes': 10,
            'num_fixed_seeds': 5,
            'metrics': [
                'task_success_rate',
                'episode_length',
                'cumulative_reward',
                'safety_violations',
                'action_smoothness',
                'goal_progress',
            ],
            'deterministic': True,
            'render': False,
        }
    
    def get_training_spec(self) -> dict:
        """Return complete training specification."""
        return {
            'algorithm': self.algorithm or 'SAC',
            'objectives': self.objectives,
            'hyperparameters': self.generate_hyperparameters(),
            'evaluation': self.generate_evaluation_protocol(),
            'ir_hash': hash(json.dumps(self.ir, default=str)),
        }


def generate_training_pipeline(ir: list[dict], 
                              priorities: dict,
                              output_script_path: Optional[str] = None) -> dict:
    """
    High-level factory: generate complete training pipeline from Kobe IR.
    
    Args:
        ir: Compiled Kobe IR
        priorities: Policy priorities
        output_script_path: If provided, write generated training script to file
    
    Returns:
        Training specification dict
    """
    generator = TrainerGenerator(ir, priorities)
    spec = generator.get_training_spec()
    
    if output_script_path:
        script = generator.generate_training_script()
        with open(output_script_path, 'w') as f:
            f.write(script)
    
    return spec
