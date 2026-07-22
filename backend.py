# backend.py  (rewritten)
from flask import Flask, request, jsonify
from flask_cors import CORS
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

app = Flask(__name__)
CORS(app)

# ── IR Interpreter ──────────────────────────────────────────────

class IRInterpreter:
    """
    Executes Kobe IR against a simulated robot state.
    The RL agent picks actions. The IR is the program structure.
    """
    def __init__(self, ir: list[dict], priorities: dict):
        self.ir         = ir
        self.priorities = priorities
        self.reset()

    def reset(self):
        self.pc               = 0
        self.loop_counters    = []    # stack of (remaining_count, start_pc)
        self.distance         = random.uniform(50, 200)
        self.colour           = random.choice(['red','green','blue','none'])
        self.ir_detected      = random.random() < 0.3
        self.uv_index         = random.uniform(0, 11)
        self.touch_pressed    = random.random() < 0.2
        self.gyro_tilt        = random.uniform(-90, 90)
        self.sound_db         = random.uniform(30, 90)
        self.speed            = 0.0
        self.collisions       = 0
        self.distance_covered = 0.0
        self.jerk             = 0.0
        self.sensor_readings  = {}

    def step(self, action: int) -> dict:
        """
        Execute one IR instruction. The agent's action modulates
        how movement instructions are carried out.
        Action 0 = cautious, 1 = normal, 2 = aggressive
        """
        if self.pc >= len(self.ir):
            return {'done': True}

        instr = self.ir[self.pc]
        done  = False
        op    = instr['op']

        if op == 'POLICY':
            self.pc += 1

        elif op == 'HALT':
            done = True

        elif op == 'WALK':
            base  = instr['speedMultiplier'] * [0.3, 0.5, 0.7][action]
            self.speed             = base
            self.distance_covered += base
            self.jerk             += base * 0.1
            if action == 2 and random.random() < 0.05:
                self.collisions += 1
            self.pc += 1

        elif op == 'RUN':
            base  = instr['speedMultiplier'] * [0.4, 0.8, 1.0][action]
            self.speed             = base
            self.distance_covered += base
            self.jerk             += base * 0.3
            if action == 2 and random.random() < 0.15:
                self.collisions += 1
            self.pc += 1

        elif op == 'STOP':
            self.speed  = 0.0
            self.jerk  += 0.05
            self.pc    += 1

        elif op == 'TURN':
            self.jerk  += 0.2
            self.pc    += 1

        elif op == 'WAIT':
            self.pc += 1

        elif op == 'SENSE':
            # Refresh sensor readings
            for sensor in instr['sensors']:
                self.sensor_readings[sensor] = self._read_sensor(sensor)
            self.pc += 1

        elif op in ('CMP_DIST','CMP_COLOUR','CMP_TOUCH',
                    'CMP_IR','CMP_UV','CMP_GYRO','CMP_SOUND'):
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

    _condition_stack: list = []

    def _read_sensor(self, sensor: str):
        if sensor == 'dist':    return self.distance
        if sensor == 'colour':  return self.colour
        if sensor == 'IR':      return self.ir_detected
        if sensor == 'UV':      return self.uv_index
        if sensor == 'touch':   return self.touch_pressed
        if sensor == 'gyro':    return self.gyro_tilt
        if sensor == 'sound':   return self.sound_db

    def _eval_cmp(self, instr: dict) -> bool:
        op = instr['op']

        if op == 'CMP_DIST':
            dist = self.sensor_readings.get('dist', self.distance)
            return _compare(dist, instr['comparator'], instr['valueCm'])

        if op == 'CMP_COLOUR':
            colour = self.sensor_readings.get('colour', self.colour)
            match  = (colour == instr['colour']) if isinstance(instr['colour'], str) else True
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
            1.0 if self.colour == 'red'   else 0.0,
            1.0 if self.colour == 'green' else 0.0,
            1.0 if self.colour == 'blue'  else 0.0,
            1.0 if self.ir_detected else 0.0,
            self.uv_index / 11.0,
            1.0 if self.touch_pressed else 0.0,
            self.gyro_tilt / 180.0,
            self.sound_db / 194.0,
            self.priorities['efficiency'],
            self.priorities['safety'],
            self.priorities['comfort'],
            self.priorities['curiosity'],
            min(self.pc / max(len(self.ir), 1), 1.0)
        ], dtype=np.float32)

    def compute_reward(self) -> float:
        p = self.priorities
        return (
            p['efficiency'] * self.distance_covered * 10.0
            - p['safety']   * self.collisions       * 500.0
            - p['comfort']  * self.jerk              * 50.0
        )


def _compare(a, op: str, b) -> bool:
    return {
        '<':  a <  b,
        '<=': a <= b,
        '>':  a >  b,
        '>=': a >= b,
        '==': a == b,
        '!=': a != b,
    }.get(op, False)


# ── Gymnasium Environment ────────────────────────────────────────

class KobeEnv(gym.Env):
    def __init__(self, ir: list[dict], priorities: dict):
        super().__init__()
        self.ir_program     = ir
        self.priorities     = priorities
        self.interpreter    = IRInterpreter(ir, priorities)
        self.action_space   = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.interpreter.reset()
        return self.interpreter.get_obs(), {}

    def step(self, action):
        result  = self.interpreter.step(int(action))
        obs     = self.interpreter.get_obs()
        reward  = self.interpreter.compute_reward()
        done    = result['done']
        return obs, reward, done, False, {}


# ── Training API ─────────────────────────────────────────────────

TRIAL_STEPS = {1: 5_000, 2: 20_000, 3: 50_000, 4: 200_000}

@app.route('/api/train', methods=['POST'])
def train():
    try:
        data        = request.json
        ir          = data['ir']
        priorities  = data['priorities']
        trial_level = data.get('trialLevel', 2)
        total_steps = TRIAL_STEPS.get(trial_level, 20_000)

        env = DummyVecEnv([lambda: KobeEnv(ir, priorities)])

        model = PPO(
            policy      = 'MlpPolicy',
            env         = env,
            verbose     = 0,
            learning_rate = 3e-4,
            n_steps     = 512,
            batch_size  = 64,
            n_epochs    = 10,
            ent_coef    = priorities.get('curiosity', 0.3) * 0.1,
        )
        model.learn(total_timesteps=total_steps)

        metrics = _evaluate(model, env)

        return jsonify({
            'success':       True,
            'metrics':       metrics,
            'trainingSteps': total_steps,
            'priorities':    priorities
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def _evaluate(model, env, episodes=50) -> dict:
    speed, safety, comfort = [], [], []

    for _ in range(episodes):
        obs  = env.reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            steps += 1

            interp = env.envs[0].interpreter
            speed.append(interp.distance_covered)
            safety.append(interp.collisions == 0)
            comfort.append(interp.jerk < 0.3)

    return {
        'speed':       round(sum(speed)  / len(speed)  * 100, 1),
        'safety':      round(sum(safety) / len(safety) * 100, 1),
        'convenience': round(sum(comfort)/ len(comfort)* 100, 1),
    }


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)