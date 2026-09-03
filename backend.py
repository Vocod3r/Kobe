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
from torch.distributions import Normal
from gymnasium import spaces

from codegen import SLIDER_MAP

logger = logging.getLogger(__name__)

TRIAL_STEPS = {1: 5_000, 2: 20_000, 3: 50_000, 4: 200_000}

# SAC Gaussian-policy log-std bounds (CleanRL sac_continuous_action.py).
SAC_LOG_STD_MAX = 2
SAC_LOG_STD_MIN = -5


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
        safety = p.get('safety', 0.5)
        # Safety slider unconditionally attenuates the speed reward on every step.
        # At safety=0 this reduces to speed_weight * distance_covered (original).
        # At safety=1 the speed reward is zeroed out entirely.
        # The collision_penalty term is retained unchanged — actual collisions
        # remain catastrophic regardless of how the speed reward is scaled.
        return (
            speed_weight * (1.0 - safety) * self.distance_covered
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


class SACActor(nn.Module):
    """Stochastic diagonal-Gaussian policy head for SAC (CleanRL conventions).

    Replaces the deterministic sigmoid ``Actor`` for SAC only.  TD3 and DroQ
    keep the deterministic ``Actor`` unchanged.

    ``forward()`` returns ``(mean, log_std)``; ``log_std`` is squashed into
    ``[SAC_LOG_STD_MIN, SAC_LOG_STD_MAX]`` with ``tanh`` (the SpinUp /
    Denis-Yarats bound used by CleanRL).

    ``get_action()`` samples with the **reparameterization trick** (``rsample``,
    not ``sample``, so gradients flow through the sample into the mean/log-std
    network) and applies tanh squashing followed by an affine rescale onto the
    environment action bounds:

        u ~ N(mean, std)                       # rsample: u = mean + std * eps
        y = tanh(u)
        a = y * action_scale + action_bias     # Kobe: scale = bias = 0.5 -> [0, 1]

    Tanh-squash log-probability correction.  For the invertible map
    a = g(u) = scale*tanh(u) + bias we have |da/du| = scale*(1 - tanh(u)^2),
    so the change-of-variables formula gives

        log pi(a|s) = log N(u; mean, std) - log( scale * (1 - y^2) + 1e-6 )

    exactly as CleanRL sac_continuous_action.py writes it ("Enforcing Action
    Bound").  For action_scale = 1 this reduces to the SAC-paper appendix form
    log pi = log N(u) - log(1 - y^2); the +1e-6 guards log(0) at y = ±1.  The
    leading minus sign is correct because the squashing map *compresses* mass
    near the boundaries (|da/du| < 1 away from u = 0), which *raises* the
    density of the squashed action relative to the raw Gaussian.

    Returns ``(action, log_prob, mean_action)``: the sampled environment-space
    action, its scalar log-probability (summed over action dims, shape (B, 1)),
    and the deterministic mode action ``tanh(mean)*scale + bias`` used for
    evaluation.
    """

    def __init__(self, obs_dim: int, hidden: int = 128, action_dim: int = 1,
                 action_scale: float = 0.5, action_bias: float = 0.5):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mean = nn.Linear(hidden, action_dim)
        self.fc_logstd = nn.Linear(hidden, action_dim)
        self.action_dim = action_dim
        # Action rescaling buffers (CleanRL).
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer('action_bias', torch.tensor(action_bias, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = SAC_LOG_STD_MIN + 0.5 * (SAC_LOG_STD_MAX - SAC_LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        u = normal.rsample()                                   # reparameterized sample
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u)
        # Enforcing Action Bound: tanh-squash + rescale change-of-variables.
        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


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
                    if isinstance(actor, SACActor):
                        # SAC is evaluated with its deterministic mode action
                        # (tanh-squashed mean), the paper's exploitation policy.
                        _, _, action = actor.get_action(t_obs)
                    else:
                        action = actor(t_obs)
                    action = action.cpu().numpy()[0]
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
    seed: int | None = None,
) -> dict:
    """
    Train a policy in the simulator.
    Hyperparameters come from codegen.SLIDER_MAP — same mapping as exported code.

    ``seed`` (optional) makes a run reproducible: when provided, the global
    random / numpy / torch RNGs are seeded before the environment and the
    networks are created, exactly like CleanRL's seeding block.  The IDE path
    (train_ipc.py without a seed) is unchanged.
    """
    alg = algorithm if algorithm in SLIDER_MAP else 'SAC'
    total_steps = TRIAL_STEPS.get(trial_level, 20_000)
    smap = SLIDER_MAP[alg]

    if seed is not None:
        # CleanRL seeding block — makes seed-controlled sweeps reproducible.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

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

    curiosity = priorities.get('curiosity', 0.3)
    comfort = priorities.get('comfort', 0.5)

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

    if alg == 'SAC':
        # ── Learned entropy temperature (CleanRL SAC autotune) ──────────────────
        #
        # DECISION (explicit): the old ``alpha = smap['curiosity']['formula'](curiosity)``
        # heuristic set a *static* temperature.  With a learned temperature the
        # curiosity slider no longer *is* alpha; instead it scales the
        # **target entropy** that the temperature is pulled toward:
        #
        #     target_entropy = -action_dim * (1.5 - curiosity)
        #
        # Higher curiosity -> less-negative target entropy -> the policy is
        # constrained to keep MORE entropy -> more persistent exploration.
        # The mapping is anchored so the slider midpoint (curiosity = 0.5, the
        # value used by every Kobe sweep) reproduces the textbook SAC default
        # target entropy of -action_dim = -1; the range over curiosity in
        # [0, 1] is [-1.5, -0.5] per action dimension.
        target_entropy = -float(env.action_space.shape[0]) * (1.5 - float(np.clip(curiosity, 0.0, 1.0)))
        # CleanRL autotune initialisation: log_alpha = 0  ->  alpha = exp(0) = 1.
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp()
        alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

        # Stochastic Gaussian actor (CleanRL conventions).  Replaces the
        # deterministic sigmoid ``Actor`` for SAC only.
        a_low = float(env.action_space.low[0])
        a_high = float(env.action_space.high[0])
        actor = SACActor(
            obs_dim,
            action_dim=int(env.action_space.shape[0]),
            action_scale=(a_high - a_low) / 2.0,
            action_bias=(a_high + a_low) / 2.0,
        ).to(device)
    else:
        # TD3 / DroQ keep the deterministic actor and the static slider alpha
        # exactly as before (NOT touched in this pass).
        actor = Actor(obs_dim).to(device)
        log_alpha = None
        alpha_opt = None
        target_entropy = None
        alpha = smap['curiosity']['formula'](curiosity)

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
        # SAC keeps Kobe's existing SINGLE critic (no twin critics in this pass).
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
    # CleanRL SAC collects uniform-random actions for the first
    # ``learning_starts`` steps to fill the replay buffer.  Scaled down to
    # Kobe's much shorter runs (10% of the budget, capped at 1000).
    sac_learning_starts = min(1000, max(100, total_steps // 10)) if alg == 'SAC' else 0

    for step in range(1, total_steps + 1):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            if alg == 'SAC':
                # CleanRL SAC data collection: uniform-random actions before
                # learning starts, then reparameterized policy samples.
                if step <= sac_learning_starts:
                    action = np.random.uniform(a_low, a_high, size=(1,)).astype(np.float32)
                else:
                    action, _, _ = actor.get_action(obs_t)
                    action = action.cpu().numpy()[0]
            else:
                action = actor(obs_t).cpu().numpy()[0]
                if alg == 'TD3':
                    action = np.clip(action + np.random.normal(0, alpha * 0.1, size=action.shape), 0.0, 1.0)

        next_obs, reward, done, _, _ = env.step(action)
        replay.append((obs.copy(), action.copy(), reward, next_obs.copy(), float(done)))
        episode_reward += reward
        obs = next_obs if not done else env.reset()[0]

        if len(replay) >= batch_size and not (alg == 'SAC' and step <= sac_learning_starts):
            utd = 4 if alg == 'DroQ' else 1
            for _ in range(utd):
                batch = random.sample(replay, batch_size)
                b_obs = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=device)
                b_act = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=device)
                b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)
                b_nobs = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=device)
                b_done = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

                if alg == 'SAC':
                    # ── SAC critic update (single Q) ──────────────────────────
                    # TD target carries the REAL maximum-entropy bonus:
                    #   y = r + gamma*(1-done) * (Q_t(s', a') - alpha*log pi(a'|s'))
                    # with a' ~ pi(·|s') via rsample under no_grad — CleanRL form.
                    with torch.no_grad():
                        next_action, next_log_prob, _ = actor.get_action(b_nobs)
                        target_next_q = target_critic(b_nobs, next_action) - alpha * next_log_prob
                        target_q = b_rew + gamma * (1.0 - b_done) * target_next_q

                    q = critic(b_obs, b_act)
                    critic_loss = F.mse_loss(q, target_q)
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    critic_opt.step()

                    # ── SAC actor update ──────────────────────────────────────
                    #   J_pi = E[ alpha * log_pi(a|s) - Q(s, a) ],  a ~ pi(·|s)
                    # The action is reparameterized (rsample) so dJ/dθ flows
                    # through both Q and log_pi.  This REPLACES the old fake
                    # entropy hack `alpha * 0.01 * mean(a * (1 - a))`.
                    pi_action, log_pi, _ = actor.get_action(b_obs)
                    q_pi = critic(b_obs, pi_action)
                    actor_loss = (alpha.detach() * log_pi - q_pi).mean()
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # ── Learned temperature (CleanRL SAC autotune) ────────────
                    #   J_alpha = E[ -alpha * (log_pi + target_entropy) ]
                    with torch.no_grad():
                        _, log_pi_for_alpha, _ = actor.get_action(b_obs)
                    alpha_loss = -(alpha * (log_pi_for_alpha + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()
                    alpha = log_alpha.exp()   # refresh tensor for next update
                else:
                    # TD3 / DroQ — byte-for-byte the pre-existing deterministic path.
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
                        if alg == 'DroQ':
                            # DroQ keeps its original variance-penalty hack
                            # (NOT touched in this SAC-only pass).
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
            msg = {
                'type': 'progress',
                'step': step,
                'total': total_steps,
                'reward': round(episode_reward, 3),
            }
            if alg == 'SAC':
                with torch.no_grad():
                    msg['logAlpha'] = round(float(log_alpha.detach().cpu()), 4)
                    msg['alpha'] = round(float(alpha.detach().cpu()), 4)
            progress_callback(msg)
            episode_reward = 0.0

    metrics = _evaluate(env, actor)
    result = {
        'success': True,
        'metrics': metrics,
        'trainingSteps': total_steps,
        'algorithm': alg,
        'priorities': priorities,
    }
    if alg == 'SAC':
        with torch.no_grad():
            result['logAlpha'] = float(log_alpha.detach().cpu())
            result['alpha'] = float(alpha.detach().cpu())
            result['targetEntropy'] = float(target_entropy)
    return result


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
