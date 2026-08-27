"""
Study tasks for the human user experiment.

Three matched tasks testing different abstraction boundaries:
T1: Obstacle avoidance (observation + action + safety)
T2: Target following (goal + observation + objective)
T3: Efficiency/smoothness tradeoff (multiple objectives + termination)
"""

from __future__ import annotations


# ── Task 1: Obstacle Avoidance ──
TASK_1_KOBE = """
hardware {
  sensors: [dist@1]
}

policy {
  safety = 0.9;
}

observe(dist) {
  dist < 20 cm then {
    stop;
  }
  else {
    walk forward;
  }
}
"""

TASK_1_PYTHON_SCAFFOLD = """
import gymnasium as gym
import numpy as np

class ObstacleAvoidanceEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Define your observation and action spaces here
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        # TODO: Initialize robot state, return initial observation
        pass
    
    def step(self, action):
        # TODO: Take action, update state
        # TODO: Check if distance < 20cm (safety constraint)
        # TODO: Return observation, reward, done, truncated, info
        pass
"""

TASK_1_DESCRIPTION = {
    'name': 'Obstacle Avoidance',
    'concept': 'Robot moves forward until it detects an obstacle, then stops.',
    'difficulty': 'Introductory',
    'key_abstractions': [
        'Observation (sensor reading)',
        'Action (walk vs stop)',
        'Safety constraint (hard stop)',
    ],
    'expected_behavior': 'Robot walks until distance < 20cm, then stops',
    'success_criteria': 'Robot successfully stops within 5cm of obstacle',
    'time_estimate_minutes': '10-15',
}


# ── Task 2: Target Following ──
TASK_2_KOBE = """
hardware {
  sensors: [dist@1, colour@2]
}

policy {
  efficiency = 0.7;
  safety = 0.8;
}

observe(dist, colour) {
  colour is red then {
    stop;
  }
  else if (dist > 50 cm) then {
    walk forward;
  }
  else {
    walk forward slowly;
  }
}
"""

TASK_2_PYTHON_SCAFFOLD = """
import gymnasium as gym
import numpy as np

class TargetFollowingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            'dist': gym.spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32),
            'colour': gym.spaces.Discrete(4),  # red, green, blue, none
        })
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        # TODO: Place target (red region) at distance > 50cm
        # TODO: Return initial observation
        pass
    
    def step(self, action):
        # TODO: Move robot forward
        # TODO: Update distance and colour reading
        # TODO: Compute reward (efficiency + safety)
        # TODO: Check termination (reached red region or max steps)
        pass
"""

TASK_2_DESCRIPTION = {
    'name': 'Target Following',
    'concept': 'Robot follows a coloured target, moving faster when far and slower when close.',
    'difficulty': 'Intermediate',
    'key_abstractions': [
        'Multiple observations',
        'Conditional movement (if-else)',
        'Objective optimization (efficiency)',
        'Multiple safety constraints',
    ],
    'expected_behavior': 'Robot walks toward red region, adjusts speed based on distance',
    'success_criteria': 'Robot reaches red region in < 60 steps',
    'time_estimate_minutes': '20-25',
}


# ── Task 3: Efficiency vs Smoothness Tradeoff ──
TASK_3_KOBE = """
hardware {
  sensors: [dist@1]
}

policy {
  efficiency = 0.6;
  comfort = 0.6;
  safety = 0.9;
}

loop until (dist < 10 cm) {
  observe(dist) {
  }
  walk forward;
}
stop;
"""

TASK_3_PYTHON_SCAFFOLD = """
import gymnasium as gym
import numpy as np

class EfficiencySmoothnessEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=200, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        # TODO: Place robot far from target
        pass
    
    def step(self, action):
        # TODO: Apply action (aggressiveness in [0, 1])
        # TODO: Robot moves forward
        # TODO: Compute two competing rewards:
        #       - Efficiency: reward for progress toward goal
        #       - Comfort: penalty for sudden action changes (jerk)
        # TODO: Combine with weights: 0.6*efficiency + 0.6*comfort
        # TODO: Termination: distance < 10cm or max steps
        pass
"""

TASK_3_DESCRIPTION = {
    'name': 'Efficiency vs Smoothness Tradeoff',
    'concept': 'Robot balances moving quickly toward a target with moving smoothly.',
    'difficulty': 'Advanced',
    'key_abstractions': [
        'Multi-objective optimization',
        'Loop with dynamic termination',
        'Action-dependent jerk penalty',
        'Competing objectives',
    ],
    'expected_behavior': 'Robot converges on smooth, reasonably-fast trajectory to target',
    'success_criteria': 'Robot reaches target with < 50% mean action change',
    'time_estimate_minutes': '25-30',
}


# ── Matched evaluation rubric ──
EVALUATION_RUBRIC = {
    'task_success': {
        'weight': 40,
        'criteria': [
            'Task completed within specified steps/time',
            'Success condition met (safety/goal)',
            'No undefined behavior',
        ],
    },
    'code_quality': {
        'weight': 20,
        'criteria': [
            'Correct observation/action/reward semantics',
            'Proper termination condition',
            'Safety constraints respected',
        ],
    },
    'understanding': {
        'weight': 20,
        'criteria': [
            'Can explain what each part does',
            'Can predict effect of changing parameters',
            'Can identify where objective matters',
        ],
    },
    'efficiency': {
        'weight': 20,
        'criteria': [
            'Minimal implementation time',
            'Few debugging cycles',
            'Clear mental model',
        ],
    },
}


# ── Transfer task (unseen) ──
TRANSFER_TASK_KOBE = """
hardware {
  sensors: [dist@1, touch@2]
}

policy {
  safety = 0.85;
  efficiency = 0.6;
}

loop until (touch) {
  observe(dist, touch) {
  }
  if (dist < 30 cm) then {
    walk forward slowly;
  }
  else {
    walk forward;
  }
}
stop;
"""

TRANSFER_TASK_DESCRIPTION = {
    'name': 'Transfer: Obstacle Detection via Touch',
    'concept': 'Robot moves toward a wall, using touch sensor as primary termination.',
    'difficulty': 'Advanced (transfers from T1 + T2)',
    'key_abstractions': [
        'Multi-sensor observation',
        'Complex conditional logic',
        'Touch-based vs distance-based detection',
    ],
    'expected_behavior': 'Robot walks steadily toward wall, stops on contact',
    'success_criteria': 'Robot touches wall at low speed (< 0.5m/s equivalent)',
    'time_estimate_minutes': '20-25',
}


# ── Registry ──
STUDY_TASKS = {
    'T1_obstacle': {
        'kobe': TASK_1_KOBE,
        'python_scaffold': TASK_1_PYTHON_SCAFFOLD,
        'description': TASK_1_DESCRIPTION,
    },
    'T2_target': {
        'kobe': TASK_2_KOBE,
        'python_scaffold': TASK_2_PYTHON_SCAFFOLD,
        'description': TASK_2_DESCRIPTION,
    },
    'T3_tradeoff': {
        'kobe': TASK_3_KOBE,
        'python_scaffold': TASK_3_PYTHON_SCAFFOLD,
        'description': TASK_3_DESCRIPTION,
    },
    'transfer': {
        'kobe': TRANSFER_TASK_KOBE,
        'description': TRANSFER_TASK_DESCRIPTION,
    },
}


def get_task(task_id: str) -> dict:
    """Retrieve a study task by ID."""
    return STUDY_TASKS.get(task_id)


def get_task_description(task_id: str) -> str:
    """Get human-readable task description."""
    task = get_task(task_id)
    if not task:
        return f"Unknown task: {task_id}"
    
    desc = task['description']
    return f"""
Task: {desc['name']}
Description: {desc['concept']}
Difficulty: {desc['difficulty']}

Key Concepts:
{chr(10).join(f"  - {c}" for c in desc['key_abstractions'])}

Expected Behavior: {desc['expected_behavior']}
Success Criteria: {desc['success_criteria']}

Estimated Time: {desc['time_estimate_minutes']} minutes
"""


def print_all_tasks():
    """Print all study tasks for reference."""
    for task_id in STUDY_TASKS:
        print(get_task_description(task_id))
        print("=" * 70)
