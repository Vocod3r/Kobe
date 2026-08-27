"""
Handwritten baseline: a deterministic, non-learned control policy.

This is a simple reactive controller that scales aggressiveness proportionally
with clearance (distance), without any learning. It serves as a floor for comparison:
if hand-written control achieves 95% safety, then a trained policy achieving 96%
isn't a meaningful result.

The policy:
  - Read distance continuously
  - Scale aggressiveness inversely (far = aggressive, close = cautious)
  - Dampen by safety priority
  - Hard-clamp in danger zone (< 5cm)
"""

import numpy as np
from typing import Optional
import gymnasium as gym


class HandwrittenBaseline:
    """Deterministic baseline controller without learning."""
    
    def __init__(self, 
                 safety_priority: float = 0.5,
                 danger_threshold: float = 5.0,
                 cautious_threshold: float = 20.0,
                 comfortable_distance: float = 100.0):
        """
        Initialize the baseline controller.
        
        Args:
            safety_priority: Weight on safety (0-1). Higher = more cautious.
            danger_threshold: Distance below which we hard-stop (cm).
            cautious_threshold: Distance below which we reduce speed (cm).
            comfortable_distance: Distance where we're at full aggressiveness (cm).
        """
        self.safety_priority = safety_priority
        self.danger_threshold = danger_threshold
        self.cautious_threshold = cautious_threshold
        self.comfortable_distance = comfortable_distance
    
    def get_action(self, obs: np.ndarray) -> float:
        """
        Compute action (aggressiveness in [0, 1]) from observation.
        
        Observation vector (from KobeEnv):
          0: distance / 200.0
          1-3: one-hot colour (red, green, blue)
          4: ir_detected
          5: uv_index / 11.0
          6: touch_pressed
          7: gyro_tilt / 180.0
          8: sound_db / 194.0
          9: efficiency priority
          10: safety priority
          11: comfort priority
          12: curiosity priority
          13: pc / len(ir)
        """
        # Extract distance (denormalize from [0, 1] back to cm)
        distance_normalized = obs[0]
        distance_cm = distance_normalized * 200.0
        
        # Extract safety priority from observation
        safety_priority = obs[10]
        
        # Proportional control: aggressiveness scales with clearance
        if distance_cm < self.danger_threshold:
            # Danger zone: stop completely
            aggressiveness = 0.0
        elif distance_cm < self.cautious_threshold:
            # Cautious zone: scale linearly from 0 to 0.2
            aggressiveness = 0.2 * (distance_cm - self.danger_threshold) / (self.cautious_threshold - self.danger_threshold)
        elif distance_cm < self.comfortable_distance:
            # Normal zone: scale linearly from 0.2 to full
            aggressiveness = 0.2 + 0.8 * (distance_cm - self.cautious_threshold) / (self.comfortable_distance - self.cautious_threshold)
        else:
            # Far away: full aggressiveness
            aggressiveness = 1.0
        
        # Dampen by safety priority: safety 1.0 = most conservative
        # (higher safety means lower aggressiveness)
        safety_dampening = 1.0 - (0.5 * safety_priority)
        aggressiveness = aggressiveness * safety_dampening
        
        # Clamp to [0, 1]
        aggressiveness = np.clip(aggressiveness, 0.0, 1.0)
        
        return float(aggressiveness)


def evaluate_handwritten_baseline(
    env: gym.Env,
    num_episodes: int = 10,
    max_steps: int = 1000,
) -> dict:
    """
    Evaluate handwritten baseline through the same metrics as trained policies.
    
    Returns dict with:
      - speed_score: average distance covered per episode
      - safety_score: percentage of episodes with 0 collisions
      - comfort_score: average "jerk" (smoothness metric, lower is better)
    """
    controller = HandwrittenBaseline(
        safety_priority=0.5,
        danger_threshold=5.0,
        cautious_threshold=20.0,
        comfortable_distance=100.0,
    )
    
    speed_scores = []
    safety_collisions = []
    comfort_scores = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_speed = 0.0
        episode_comfort = 0.0
        episode_collisions = 0
        
        while not done and step_count < max_steps:
            action = controller.get_action(obs)
            obs, reward, done, truncated, info = env.step(np.array([action]))
            step_count += 1
            
            # Accumulate metrics
            interpreter = env.interpreter
            episode_speed = interpreter.distance_covered
            episode_comfort = interpreter.jerk
            episode_collisions = interpreter.collisions
        
        speed_scores.append(episode_speed)
        safety_collisions.append(episode_collisions)
        comfort_scores.append(episode_comfort)
    
    # Compute aggregate metrics
    avg_speed = np.mean(speed_scores)
    safety_percentage = 100.0 * (sum(1 for c in safety_collisions if c == 0) / len(safety_collisions))
    avg_comfort = np.mean(comfort_scores)
    
    return {
        'speed_score': float(avg_speed),
        'safety_score': float(safety_percentage),
        'comfort_score': float(avg_comfort),
        'episodes': num_episodes,
        'episode_speeds': speed_scores,
        'episode_collisions': safety_collisions,
        'episode_comforts': comfort_scores,
    }
