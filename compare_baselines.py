"""
Comparison tool: evaluate handwritten baseline vs trained policies.

This script runs the handwritten baseline and optionally trained policies
through the same evaluation harness, enabling direct comparison of:
  - Speed (average distance covered)
  - Safety (collision rate)
  - Comfort (smoothness)
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from backend import KobeEnv
from handwritten_baseline import evaluate_handwritten_baseline
from lexer import tokenize
from parser import Parser
from compiler import compile
from priorities import DEFAULTS

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_test_program():
    """Create a simple test program for evaluation."""
    code = """
    hardware {
      sensors: [dist@1, touch@2, IR@3]
    }
    
    policy {
      safety = 0.8;
      efficiency = 0.7;
    }
    
    loop (3) {
      observe(dist) {
        dist < 30 cm then {
          stop;
          break;
        }
      }
      walk forward;
    }
    """
    
    tokens = tokenize(code)
    parser = Parser(tokens)
    ast = parser.parse()
    ir = compile(ast, DEFAULTS)
    return ir


def evaluate_policy(env, actor=None, num_episodes=10, max_steps=1000, policy_name="Policy"):
    """
    Evaluate a policy (or random if actor is None).
    
    Returns dict with speed, safety, comfort scores.
    """
    speed_scores = []
    collision_counts = []
    comfort_scores = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if actor is None:
                # Random policy
                action = np.random.uniform(0, 1, (1,))
            else:
                # Use provided actor
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action_tensor = actor(obs_tensor)
                action = action_tensor.cpu().numpy()[0]
            
            obs, reward, done, _, _ = env.step(action)
            steps += 1
        
        # Get final metrics from interpreter
        interpreter = env.interpreter
        speed_scores.append(interpreter.distance_covered)
        collision_counts.append(interpreter.collisions)
        comfort_scores.append(interpreter.jerk)
    
    return {
        'name': policy_name,
        'speed': float(np.mean(speed_scores)),
        'safety': float(100.0 * (sum(1 for c in collision_counts if c == 0) / len(collision_counts))),
        'comfort': float(np.mean(comfort_scores)),
        'episodes': num_episodes,
    }


def compare_baselines(include_trained=False, trained_actor_path=None):
    """
    Compare random, handwritten, and optionally trained baselines.
    
    Returns list of dicts with evaluation results.
    """
    print("=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    # Create test program and environment
    ir = create_test_program()
    env = KobeEnv(ir, DEFAULTS, algorithm='SAC')
    
    results = []
    
    # Random baseline
    print("\n[1/3] Evaluating random policy...")
    random_results = evaluate_policy(env, actor=None, num_episodes=5, policy_name="Random")
    results.append(random_results)
    print(f"  Speed: {random_results['speed']:.1f} | Safety: {random_results['safety']:.1f}% | Comfort: {random_results['comfort']:.2f}")
    
    # Handwritten baseline
    print("[2/3] Evaluating handwritten baseline...")
    handwritten_results = evaluate_handwritten_baseline(env, num_episodes=5)
    results.append({
        'name': 'Handwritten',
        'speed': handwritten_results['speed_score'],
        'safety': handwritten_results['safety_score'],
        'comfort': handwritten_results['comfort_score'],
        'episodes': handwritten_results['episodes'],
    })
    print(f"  Speed: {results[-1]['speed']:.1f} | Safety: {results[-1]['safety']:.1f}% | Comfort: {results[-1]['comfort']:.2f}")
    
    # Trained baseline (if available)
    if include_trained and TORCH_AVAILABLE and trained_actor_path:
        print("[3/3] Evaluating trained SAC policy...")
        try:
            actor = torch.load(trained_actor_path)
            trained_results = evaluate_policy(env, actor=actor, num_episodes=5, policy_name="SAC (Trained)")
            results.append(trained_results)
            print(f"  Speed: {trained_results['speed']:.1f} | Safety: {trained_results['safety']:.1f}% | Comfort: {trained_results['comfort']:.2f}")
        except Exception as e:
            print(f"  ERROR loading trained actor: {e}")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<20} {'Speed':<12} {'Safety':<12} {'Comfort':<12}")
    print("-" * 60)
    for result in results:
        print(f"{result['name']:<20} {result['speed']:>10.1f}   {result['safety']:>10.1f}%  {result['comfort']:>10.2f}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baseline policies')
    parser.add_argument('--trained', action='store_true', help='Include trained policy comparison')
    parser.add_argument('--actor-path', type=str, help='Path to trained actor checkpoint')
    
    args = parser.parse_args()
    
    results = compare_baselines(
        include_trained=args.trained,
        trained_actor_path=args.actor_path
    )
