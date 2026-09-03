# backend.py — Kobe live-training backend (simulator + optional Flask API)
from __future__ import annotations

import logging
import random
import sys
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from codegen import SLIDER_MAP

logger = logging.getLogger(__name__)

TRIAL_STEPS = {1: 5_000, 2: 20_000, 3: 50_000, 4: 200_000}


# ── IR Interpreter ────────────────────────────────────────────────────────────


class IRInterpreter:
    """Executes Kobe IR against a simulated robot state."""

    _condition_stack: list = []

    def __init__(self, ir: list[dict], priorities: dict):
        self.ir = ir
        self.priorities = priorities
        self.reset()

    def reset(self):
        self.pc = 0
        self.loop_counters: list[list] = []
        self.distance = random.uniform(50, 200)
        self.colour = random.choice(['red', 'green', 'blue', 'none'])
        self.ir_detected = random.random() < 0.3
        self.uv_index = random.uniform(0, 11)
        self.touch_pressed = random.random() < 0.2
        self.gyro_tilt = random.uniform(-90, 90)
        self.sound_db = random.uniform(30, 90)
        self.speed = 0.0
        self.collisions = 0
        self.distance_covered = 0.0
        self.jerk = 0.0
        self.sensor_readings: dict = {}
        IRInterpreter._condition_stack = []

    def step(self, action: float) -> dict:
        """Execute one IR instruction. Action in [0, 1] modulates speed/aggression."""
        if self.pc >= len(self.ir):
            return {'done': True}

        instr = self.ir[self.pc]
        done = False
        op = instr['op']
        aggressiveness = float(np.clip(action, 0.0, 1.0))
        speed_scale = 0.3 + aggressiveness * 0.7

        if op in ('ALGORITHM', 'HARDWARE'):
            self.pc += 1
        elif op == 'POLICY':
            self.pc += 1
        elif op == 'HALT':
            done = True
        elif op == 'WALK':
            base = instr['speedMultiplier'] * speed_scale
            self.speed = base
            self.distance_covered += base
            self.jerk += base * 0.1
            if aggressiveness > 0.75 and random.random() < 0.05:
                self.collisions += 1
            self.pc += 1
        elif op == 'RUN':
            base = instr['speedMultiplier'] * speed_scale * 1.4
            self.speed = base
            self.distance_covered += base
            self.jerk += base * 0.3
            if aggressiveness > 0.75 and random.random() < 0.15:
                self.collisions += 1
            self.pc += 1
        elif op == 'STOP':
            self.speed = 0.0
            self.jerk += 0.05
            self.pc += 1
        elif op == 'TURN':
            self.jerk += 0.2
            self.pc += 1
        elif op == 'WAIT':
            self.pc += 1
        elif op == 'SENSE':
            for sensor in instr['sensors']:
                self.sensor_readings[sensor] = self._read_sensor(sensor)
            self.pc += 1
        elif op in ('CMP_DIST', 'CMP_COLOUR', 'CMP_TOUCH',
                    'CMP_IR', 'CMP_UV', 'CMP_GYRO', 'CMP_SOUND'):
            self._condition_stack.append(self._eval_cmp(instr))
            self.pc += 1
        elif op == 'AND':
            b = self._condition_stack.pop()
            a = self._condition_stack.pop()
            self._condition_stack.append(a and b)
            self.pc += 1
        elif op == 'OR':
            b = self._condition_stack.pop()
            a = self._condition_stack.pop()
            self._condition_stack.append(a or b)
            self.pc += 1
        elif op == 'NOT':
            self._condition_stack.append(not self._condition_stack.pop())
            self.pc += 1
        elif op == 'JUMP_IF_FALSE':
            val = self._condition_stack.pop() if self._condition_stack else False
            self.pc = (self.pc + 1) if val else instr['target']
        elif op == 'JUMP':
            self.pc = instr['target']
        elif op == 'LOOP_START':
            self.loop_counters.append([instr['count'], self.pc])
            self.pc += 1
        elif op == 'LOOP_END':
            if self.loop_counters:
                self.loop_counters[-1][0] -= 1
                if self.loop_counters[-1][0] > 0:
                    self.pc = self.loop_counters[-1][1] + 1
                else:
                    self.loop_counters.pop()
                    self.pc += 1
        elif op == 'BREAK':
            self.pc = instr['target']
        else:
            self.pc += 1

        if self.collisions > 0:
            done = True

        return {'done': done}

    def _read_sensor(self, sensor: str):
        mapping = {
            'dist': self.distance,
            'colour': self.colour,
            'IR': self.ir_detected,
            'UV': self.uv_index,
            'touch': self.touch_pressed,
            'gyro': self.gyro_tilt,
            'sound': self.sound_db,
        }
        return mapping.get(sensor)

    def _eval_cmp(self, instr: dict) -> bool:
        op = instr['op']
        if op == 'CMP_DIST':
            dist = self.sensor_readings.get('dist', self.distance)
            return _compare(dist, instr['comparator'], instr['valueCm'])
        if op == 'CMP_COLOUR':
            colour = self.sensor_readings.get('colour', self.colour)
            match = (colour == instr['colour']) if isinstance(instr['colour'], str) else True
            return (not match) if instr['negate'] else match
        if op == 'CMP_TOUCH':
            return self.sensor_readings.get('touch', self.touch_pressed)
        if op == 'CMP_IR':
            if instr['mode'] == 'detected':
                return self.sensor_readings.get('IR', self.ir_detected)
            return _compare(self.sensor_readings.get('IR', 0), instr['comparator'], instr['value'])
        if op == 'CMP_UV':
            if instr['mode'] == 'detected':
                return self.uv_index > 0
            return _compare(self.sensor_readings.get('UV', self.uv_index), instr['comparator'], instr['index'])
        if op == 'CMP_GYRO':
            return _compare(self.sensor_readings.get('gyro', self.gyro_tilt), instr['comparator'], instr['degrees'])
        if op == 'CMP_SOUND':
            return _compare(self.sensor_readings.get('sound', self.sound_db), instr['comparator'], instr['db'])
        return False

    def get_obs(self) -> np.ndarray:
        return np.array([
            self.distance / 200.0,
            1.0 if self.colour == 'red' else 0.0,
            1.0 if self.colour == 'green' else 0.0,
            1.0 if self.colour == 'blue' else 0.0,
            1.0 if self.ir_detected else 0.0,
            self.uv_index / 11.0,
            1.0 if self.touch_pressed else 0.0,
            self.gyro_tilt / 180.0,
            self.sound_db / 194.0,
            self.priorities['efficiency'],
            self.priorities['safety'],
            self.priorities['comfort'],
            self.priorities['curiosity'],
            min(self.pc / max(len(self.ir), 1), 1.0),
        ], dtype=np.float32)

    def compute_reward(self) -> float:
        p = self.priorities
        alg = getattr(self, 'algorithm', 'SAC')
        smap = SLIDER_MAP.get(alg, SLIDER_MAP['SAC'])
        collision_penalty = smap['safety']['formula'](p.get('safety', 0.5))
        speed_weight = smap['efficiency']['formula'](p.get('efficiency', 0.5))
        jerk_weight = p.get('comfort', 0.5) * 50.0
        return (
            speed_weight * self.distance_covered
            - collision_penalty * self.collisions
            - jerk_weight * self.jerk
        )


def _compare(a, op: str, b) -> bool:
    return {
        '<': a < b, '<=': a <= b, '>': a > b,
        '>=': a >= b, '==': a == b, '!=': a != b,
    }.get(op, False)


# ── Gymnasium Environment ─────────────────────────────────────────────────────


class KobeEnv(gym.Env):
    def __init__(self, ir: list[dict], priorities: dict, algorithm: str = 'SAC'):
        super().__init__()
        self.ir_program = ir
        self.priorities = priorities
        self.algorithm = algorithm
        self.interpreter = IRInterpreter(ir, priorities)
        self.interpreter.algorithm = algorithm
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(14,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.interpreter.reset()
        self.interpreter.algorithm = self.algorithm
        return self.interpreter.get_obs(), {}

    def step(self, action):
        a = float(action[0]) if hasattr(action, '__len__') else float(action)
        result = self.interpreter.step(a)
        obs = self.interpreter.get_obs()
        reward = self.interpreter.compute_reward()
        done = result['done']
        return obs, reward, done, False, {}


# ── Policy networks ───────────────────────────────────────────────────────────


class Actor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(obs_dim + 1, hidden), nn.ReLU(),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.extend([
            nn.Linear(hidden, hidden), nn.ReLU(),
        ])
        if dropout_p > 0.0:
            layers.append(nn.Dropout(p=dropout_p))
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


# ── Training ──────────────────────────────────────────────────────────────────


def _evaluate(env: KobeEnv, actor: Actor | None, episodes: int = 50) -> dict:
    speed, safety, comfort = [], [], []
    device = next(actor.parameters()).device if actor is not None else torch.device('cpu')

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            if actor is None:
                action = np.array([random.random()], dtype=np.float32)
            else:
                with torch.no_grad():
                    t_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    action = actor(t_obs).cpu().numpy()[0]
            obs, _, done, _, _ = env.step(action)
            steps += 1
            interp = env.interpreter
            speed.append(interp.distance_covered)
            safety.append(interp.collisions == 0)
            comfort.append(interp.jerk < 0.3)

    return {
        'speed': round(sum(speed) / len(speed) * 100, 1),
        'safety': round(sum(safety) / len(safety) * 100, 1),
        'convenience': round(sum(comfort) / len(comfort) * 100, 1),
    }


def train_policy(
    ir: list[dict],
    priorities: dict,
    algorithm: str = 'SAC',
    trial_level: int = 2,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """
    Train a policy in the simulator.
    Hyperparameters come from codegen.SLIDER_MAP — same mapping as exported code.
    """
    alg = algorithm if algorithm in SLIDER_MAP else 'SAC'
    total_steps = TRIAL_STEPS.get(trial_level, 20_000)
    smap = SLIDER_MAP[alg]

    env = KobeEnv(ir, priorities, alg)
    obs_dim = env.observation_space.shape[0]

    if alg == 'random':
        metrics = _evaluate(env, None)
        return {
            'success': True,
            'metrics': metrics,
            'trainingSteps': 0,
            'algorithm': alg,
            'priorities': priorities,
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = Actor(obs_dim).to(device)

    curiosity = priorities.get('curiosity', 0.3)
    comfort = priorities.get('comfort', 0.5)

    alpha = smap['curiosity']['formula'](curiosity)
    comfort_result = smap['comfort']['formula'](comfort)
    dropout_p = 0.0
    if isinstance(comfort_result, tuple):
        soft_eps, second_val = comfort_result
        if alg == 'TD3':
            noise_clip = second_val
        elif alg == 'DroQ':
            noise_clip = 0.2
            dropout_p = float(second_val)
        else:
            noise_clip = 0.2
    else:
        soft_eps, noise_clip = comfort_result, 0.2

    if alg == 'DroQ':
        critic = Critic(obs_dim, dropout_p=dropout_p).to(device)
        target_critic = Critic(obs_dim, dropout_p=dropout_p).to(device)
        target_critic.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    elif alg == 'TD3':
        critic = Critic(obs_dim).to(device)
        target_critic = Critic(obs_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())
        critic2 = Critic(obs_dim).to(device)
        target_critic2 = Critic(obs_dim).to(device)
        target_critic2.load_state_dict(critic2.state_dict())
        critic_opt = torch.optim.Adam(list(critic.parameters()) + list(critic2.parameters()), lr=3e-4)
    else:
        critic = Critic(obs_dim).to(device)
        target_critic = Critic(obs_dim).to(device)
        target_critic.load_state_dict(critic.state_dict())
        critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)

    gamma = 0.99
    batch_size = 64
    replay: list[tuple] = []
    obs, _ = env.reset()
    episode_reward = 0.0
    log_interval = max(total_steps // 20, 500)

    for step in range(1, total_steps + 1):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action = actor(obs_t).cpu().numpy()[0]
            if alg == 'TD3':
                action = np.clip(action + np.random.normal(0, alpha * 0.1, size=action.shape), 0.0, 1.0)

        next_obs, reward, done, _, _ = env.step(action)
        replay.append((obs.copy(), action.copy(), reward, next_obs.copy(), float(done)))
        episode_reward += reward
        obs = next_obs if not done else env.reset()[0]

        if len(replay) >= batch_size:
            utd = 4 if alg == 'DroQ' else 1
            for _ in range(utd):
                batch = random.sample(replay, batch_size)
                b_obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
                b_act = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=device)
                b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
                b_done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

                with torch.no_grad():
                    next_action = actor(b_nobs)
                    if alg == 'TD3':
                        noise = torch.clamp(torch.randn_like(next_action) * noise_clip, -noise_clip, noise_clip)
                        next_action = torch.clamp(next_action + noise, 0.0, 1.0)
                        target_q1 = target_critic(b_nobs, next_action)
                        target_q2 = target_critic2(b_nobs, next_action)
                        target_next_q = torch.min(target_q1, target_q2)
                    else:
                        target_next_q = target_critic(b_nobs, next_action)
                    target_q = b_rew + gamma * (1 - b_done) * target_next_q

                q = critic(b_obs, b_act)
                if alg == 'TD3':
                    q2 = critic2(b_obs, b_act)
                    critic_loss = F.mse_loss(q, target_q) + F.mse_loss(q2, target_q)
                else:
                    critic_loss = F.mse_loss(q, target_q)

                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                if alg != 'TD3' or step % 2 == 0:
                    pred_action = actor(b_obs)
                    actor_loss = -critic(b_obs, pred_action).mean()
                    if alg in ('SAC', 'DroQ'):
                        actor_loss = actor_loss - alpha * 0.01 * torch.mean(pred_action * (1 - pred_action) + 1e-6)
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                for tp, sp in zip(target_critic.parameters(), critic.parameters()):
                    tp.data.copy_(soft_eps * tp.data + (1 - soft_eps) * sp.data)
                if alg == 'TD3':
                    for tp, sp in zip(target_critic2.parameters(), critic2.parameters()):
                        tp.data.copy_(soft_eps * tp.data + (1 - soft_eps) * sp.data)

        if progress_callback and step % log_interval == 0:
            progress_callback({
                'type': 'progress',
                'step': step,
                'total': total_steps,
                'reward': round(episode_reward, 3),
            })
            episode_reward = 0.0

    metrics = _evaluate(env, actor)
    return {
        'success': True,
        'metrics': metrics,
        'trainingSteps': total_steps,
        'algorithm': alg,
        'priorities': priorities,
    }


# ── Flask API (optional dev server) ─────────────────────────────────────────────


def create_app():
    from flask import Flask, jsonify, request
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route('/api/train', methods=['POST'])
    def train():
        try:
            data = request.json
            result = train_policy(
                ir=data['ir'],
                priorities=data['priorities'],
                algorithm=data.get('algorithm', 'SAC'),
                trial_level=data.get('trialLevel', 2),
            )
            return jsonify(result)
        except Exception as e:
            logger.exception('Training failed')
            return jsonify({'error': str(e)}), 500

    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({'status': 'ok'})

    return app


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    create_app().run(debug=True, host='0.0.0.0', port=5000)
