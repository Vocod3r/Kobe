# codegen.py — Kobe Code Generator (TorchRL native)
#
# Slider → TorchRL hyperparameter mapping is ALGORITHM-SPECIFIC.
# See SLIDER_MAP below for the full table.

from __future__ import annotations
from textwrap import dedent

# ── Slider → TorchRL mapping (per algorithm) ────────────────────────────────
#
# This is the single source of truth for what each slider does.
# The IDE reads this to show the child a live description.
# The codegen reads this to emit the correct TorchRL arguments.

SLIDER_MAP = {
    'SAC': {
        'curiosity': {
            'param':   'SACLoss alpha_init',
            'effect':  'Entropy temperature — how randomly the robot explores',
            'formula': lambda v: v,                          # direct
        },
        'comfort': {
            'param':   'SoftUpdate eps',
            'effect':  'Target network smoothing — how steadily the robot learns',
            'formula': lambda v: 0.990 + v * 0.009,         # [0.990, 0.999]
        },
        'safety': {
            'param':   'reward collision penalty',
            'effect':  'How hard the robot avoids obstacles',
            'formula': lambda v: v * 5000.0,
        },
        'efficiency': {
            'param':   'reward speed weight',
            'effect':  'How much the robot values moving fast',
            'formula': lambda v: v * 10.0,
        },
    },
    'TD3': {
        'curiosity': {
            'param':   'AdditiveGaussianWrapper sigma_init',
            'effect':  'Exploration noise — how boldly the robot tries new things',
            'formula': lambda v: 0.05 + v * 0.25,           # [0.05, 0.30]
        },
        'comfort': {
            'param':   'SoftUpdate eps + TD3Loss noise_clip',
            'effect':  'Action precision — how smooth and deliberate movements are',
            'formula': lambda v: (0.990 + v * 0.009,        # eps
                                  0.5 * (1.0 - v * 0.4)),   # noise_clip [0.50, 0.30]
        },
        'safety': {
            'param':   'reward collision penalty',
            'effect':  'How hard the robot avoids obstacles',
            'formula': lambda v: v * 5000.0,
        },
        'efficiency': {
            'param':   'reward speed weight',
            'effect':  'How much the robot values moving fast',
            'formula': lambda v: v * 10.0,
        },
    },
    'DroQ': {
        'curiosity': {
            'param':   'SACLoss alpha_init',
            'effect':  'Entropy temperature — how randomly the robot explores',
            'formula': lambda v: v,
        },
        'comfort': {
            'param':   'SoftUpdate eps + Q-network dropout probability',
            'effect':  'Q-network stability — higher comfort = more stable value estimates',
            'formula': lambda v: (0.990 + v * 0.009,        # eps
                                  max(0.001, 0.05 * (1.0 - v))),  # dropout [0.05, ~0]
        },
        'safety': {
            'param':   'reward collision penalty',
            'effect':  'How hard the robot avoids obstacles',
            'formula': lambda v: v * 5000.0,
        },
        'efficiency': {
            'param':   'reward speed weight',
            'effect':  'How much the robot values moving fast',
            'formula': lambda v: v * 10.0,
        },
    },
    'random': {
        'curiosity':  {'param': 'N/A', 'effect': 'No effect — random policy only', 'formula': lambda v: v},
        'comfort':    {'param': 'N/A', 'effect': 'No effect — random policy only', 'formula': lambda v: v},
        'safety':     {'param': 'reward collision penalty', 'effect': 'Shown in results only', 'formula': lambda v: v * 5000.0},
        'efficiency': {'param': 'reward speed weight',      'effect': 'Shown in results only', 'formula': lambda v: v * 10.0},
    },
}


# ── Sensor metadata ──────────────────────────────────────────────────────────

EV3_SENSOR = {
    'dist':   {'cls': 'UltrasonicSensor', 'read': '{v} = float({n}.distance_centimeters)',  'lo': 0.0,    'hi': 255.0},
    'colour': {'cls': 'ColorSensor',      'read': '{v} = float({n}.color)',                 'lo': 0.0,    'hi': 8.0  },
    'IR':     {'cls': 'InfraredSensor',   'read': '{v} = float({n}.proximity)',              'lo': 0.0,    'hi': 100.0},
    'touch':  {'cls': 'TouchSensor',      'read': '{v} = float({n}.is_pressed)',             'lo': 0.0,    'hi': 1.0  },
    'gyro':   {'cls': 'GyroSensor',       'read': '{v} = float({n}.angle)',                  'lo': -180.0, 'hi': 180.0},
    'sound':  {'cls': 'SoundSensor',      'read': '{v} = float({n}.sound_pressure)',         'lo': 0.0,    'hi': 100.0},
}

RPI_SENSOR = {
    'dist':   {'lo': 0.0,   'hi': 400.0, 'setup': 'GPIO.setup(TRIG, GPIO.OUT)\nGPIO.setup(ECHO, GPIO.IN)', 'read': '{v} = _read_dist()'},
    'colour': {'lo': 0.0,   'hi': 9.0,   'setup': '',                                                       'read': '{v} = _read_colour()'},
    'IR':     {'lo': 0.0,   'hi': 1.0,   'setup': 'GPIO.setup(IR_PIN, GPIO.IN)',                            'read': '{v} = float(not GPIO.input(IR_PIN))'},
    'touch':  {'lo': 0.0,   'hi': 1.0,   'setup': 'GPIO.setup(TOUCH_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)', 'read': '{v} = float(not GPIO.input(TOUCH_PIN))'},
    'gyro':   {'lo': -250.0,'hi': 250.0, 'setup': "import mpu6050\n_imu = mpu6050.mpu6050(0x68)",          'read': "{v} = _imu.get_gyro_data()['z']"},
}


# ── Public entry point ───────────────────────────────────────────────────────

def generate(
    ir:         list[dict],
    priorities: dict,
    hardware:   dict | None = None,
    algorithm:  str         = 'SAC',
) -> dict[str, str]:
    hw  = hardware or {'target': 'EV3',
                       'motors':  [{'port': 'A'}, {'port': 'B'}],
                       'sensors': [{'type': 'dist', 'port': '1'}]}
    alg = algorithm if algorithm in SLIDER_MAP else 'SAC'

    target  = hw.get('target', 'EV3').upper()
    motors  = hw.get('motors',  [{'port': 'A'}, {'port': 'B'}])
    sensors = hw.get('sensors', [{'type': 'dist', 'port': '1'}])

    safety_cm = _extract_safety_threshold(ir)

    client = (_client_ev3(motors, sensors) if target == 'EV3'
              else _client_rpi(motors, sensors))

    return {
        'client':      client,
        'environment': _environment(motors, sensors, priorities, safety_cm, alg),
        'train':       _train(alg, priorities, motors, sensors),
    }


def slider_descriptions(algorithm: str) -> dict:
    """
    Returns human-readable descriptions of what each slider does
    for the given algorithm. The IDE uses this to show live tooltips.
    """
    alg = algorithm if algorithm in SLIDER_MAP else 'SAC'
    return {
        k: {'param': v['param'], 'effect': v['effect']}
        for k, v in SLIDER_MAP[alg].items()
    }


# ── EV3 client ───────────────────────────────────────────────────────────────

def _client_ev3(motors, sensors):
    n_act   = len(motors)
    n_state = len(motors) + 2 + len(sensors)

    sensor_classes = ', '.join({EV3_SENSOR[s['type']]['cls']
                                 for s in sensors if s['type'] in EV3_SENSOR})

    motor_setup = []
    motor_vars  = []
    for i, m in enumerate(motors):
        v   = f"_m{m['port'].lower()}"
        dir = 'Direction.COUNTERCLOCKWISE' if i % 2 else 'Direction.CLOCKWISE'
        motor_setup.append(f"{v} = Motor(Port.{m['port'].upper()}, {dir})")
        motor_vars.append(v)

    sensor_setup = []
    sensor_reads = []
    sensor_vars  = []
    for s in sensors:
        t, sv = s['type'], f"_s{s['type']}"
        prt   = f"Port.S{s['port']}" if not s['port'].isalpha() else f"Port.{s['port'].upper()}"
        if t in EV3_SENSOR:
            sensor_setup.append(f"{sv} = {EV3_SENSOR[t]['cls']}({prt})")
            read = EV3_SENSOR[t]['read'].format(v=f"_r{t}", n=sv)
            sensor_reads.append(f"    {read}")
            sensor_vars.append(f"_r{t}")

    apply_lines = '\n'.join(f"    {v}.run(int(action[{i}] * 100))"
                             for i, v in enumerate(motor_vars))
    state_vals  = ([f"float({v}.angle())" for v in motor_vars]
                   + ["float(hub.imu.tilt()[0])", "float(hub.imu.tilt()[1])"]
                   + sensor_vars)

    return (
        "# Generated by Kobe — EV3 client (MicroPython/Pybricks)\n"
        "# Upload to hub before running train.py on laptop.\n"
        "import ustruct\n"
        "from micropython import kbd_intr\n"
        "from pybricks.hubs import EV3Brick\n"
        "from pybricks.parameters import Direction, Port\n"
        "from pybricks.pupdevices import Motor\n"
        "from pybricks.tools import wait\n"
        + (f"from pybricks.ev3devices import {sensor_classes}\n" if sensor_classes else "")
        + "from uselect import poll\n"
          "from usys import stdin, stdout\n\n"
          "hub = EV3Brick()\n"
        + "\n".join(motor_setup) + "\n"
        + "\n".join(sensor_setup) + "\n\n"
          "kbd_intr(-1)\n"
          "_kb = poll()\n"
          "_kb.register(stdin)\n\n"
          "while True:\n"
          "    while not _kb.poll(0):\n"
          "        wait(1)\n\n"
        + f"    action = ustruct.unpack('!{'f' * n_act}', stdin.buffer.read({n_act * 4}))\n\n"
        + apply_lines + "\n"
        + "\n".join(sensor_reads) + "\n\n"
        + f"    stdout.buffer.write(ustruct.pack('!{'f' * n_state}', {', '.join(state_vals)}))\n"
    )


# ── RPi client ───────────────────────────────────────────────────────────────

def _client_rpi(motors, sensors):
    n_act   = len(motors)
    n_state = len(motors) + len(sensors)

    motor_lines  = []
    pwm_vars     = []
    apply_lines  = []
    for i, m in enumerate(motors):
        en, in1, in2 = m.get('pin_en', 18+i*5), m.get('pin_in1', 23+i*5), m.get('pin_in2', 24+i*5)
        v = f"_pwm{i}"
        motor_lines += [f"GPIO.setup([{en},{in1},{in2}], GPIO.OUT)",
                        f"{v} = GPIO.PWM({en}, 1000); {v}.start(0)"]
        pwm_vars.append(v)
        apply_lines += [f"        GPIO.output({in1}, action[{i}]>0); GPIO.output({in2}, action[{i}]<0)",
                        f"        {v}.ChangeDutyCycle(abs(action[{i}])*100)"]

    sensor_setup = []
    sensor_reads = []
    sensor_vars  = []
    for s in sensors:
        t, sv = s['type'], f"_rs{s['type']}"
        if t in RPI_SENSOR:
            sensor_setup.append(RPI_SENSOR[t]['setup'])
            sensor_reads.append("        " + RPI_SENSOR[t]['read'].format(v=sv))
            sensor_vars.append(sv)

    state_vals = ["0.0"] * len(motors) + sensor_vars

    return (
        "# Generated by Kobe — Raspberry Pi client\n"
        "import RPi.GPIO as GPIO, struct, time, sys\n"
        "GPIO.setmode(GPIO.BCM)\n"
        + "\n".join(motor_lines) + "\n"
        + "\n".join(s for s in sensor_setup if s) + "\n\n"
          "try:\n"
          "    while True:\n"
        + f"        raw = sys.stdin.buffer.read({n_act * 4})\n"
          "        if not raw: break\n"
        + f"        action = struct.unpack('!{'f'*n_act}', raw)\n"
        + "\n".join(apply_lines) + "\n"
        + "\n".join(sensor_reads) + "\n"
        + f"        sys.stdout.buffer.write(struct.pack('!{'f'*n_state}', {', '.join(state_vals)}))\n"
          "        sys.stdout.buffer.flush()\n"
          "finally:\n"
        + "\n".join(f"    {v}.stop()" for v in pwm_vars) + "\n"
          "    GPIO.cleanup()\n"
    )


# ── TorchRL environment ──────────────────────────────────────────────────────

def _environment(motors, sensors, priorities, safety_cm, algorithm):
    """
    The reward function is the same across algorithms.
    safety and efficiency sliders always map to reward weights.
    The TorchRL-specific hyperparams (alpha, eps, etc.) live in train.py.
    """
    p          = priorities
    action_dim = len(motors)
    state_dim  = len(motors) + 2 + len(sensors)
    dist_idx   = len(motors) + 2   # after motor angles + pitch/roll

    lo_vals = ([0.0] * len(motors) + [-90.0, -90.0]
               + [_lo(s) for s in sensors])
    hi_vals = ([360.0] * len(motors) + [90.0, 90.0]
               + [_hi(s) for s in sensors])

    # safety and efficiency always go to reward — algorithm-independent
    collision_penalty = SLIDER_MAP[algorithm]['safety']['formula'](p.get('safety', 0.5))
    speed_weight      = SLIDER_MAP[algorithm]['efficiency']['formula'](p.get('efficiency', 0.5))
    jerk_weight       = p.get('comfort', 0.5) * 50.0   # comfort also affects jerk penalty

    return dedent(f"""\
        # Generated by Kobe — TorchRL environment
        # Algorithm: {algorithm}
        # DO NOT EDIT — regenerate from Kobe IDE.
        #
        # Slider → reward weight mapping for {algorithm}:
        #   safety     = {p.get('safety', 0.5)} → collision penalty = {collision_penalty:.1f}
        #   efficiency = {p.get('efficiency', 0.5)} → speed reward weight = {speed_weight:.1f}
        #   comfort    = {p.get('comfort', 0.5)} → jerk penalty weight = {jerk_weight:.1f}
        #   curiosity  = {p.get('curiosity', 0.3)} → see train.py ({SLIDER_MAP[algorithm]['curiosity']['param']})

        import torch, struct, serial, time
        from tensordict import TensorDict
        from torchrl.envs import EnvBase
        from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec


        class HardwareConnection:
            def __init__(self, port='/dev/ttyACM0', baud=115200):
                self.conn = serial.Serial(port, baud, timeout=2.0)
                time.sleep(2.0)

            def send(self, values: list):
                self.conn.write(struct.pack(f'!{{len(values)}}f', *values))

            def receive(self, n: int) -> list:
                raw = self.conn.read(n * 4)
                return list(struct.unpack(f"!{{n}}f", raw)) if len(raw) == n * 4 else [0.0] * n

            def close(self):
                self.conn.close()


        class KobeEnv(EnvBase):

            STATE_DIM  = {state_dim}
            ACTION_DIM = {action_dim}

            # Reward weights — derived from Kobe policy block sliders
            # Algorithm: {algorithm}
            _COLLISION_PENALTY = {collision_penalty:.2f}  # safety   = {p.get('safety', 0.5)}
            _SPEED_WEIGHT      = {speed_weight:.2f}       # efficiency = {p.get('efficiency', 0.5)}
            _JERK_WEIGHT       = {jerk_weight:.2f}        # comfort  = {p.get('comfort', 0.5)}
            _COLLISION_DIST_CM = {safety_cm:.1f}          # from: dist < X in program

            def __init__(self, connection: HardwareConnection, device="cpu"):
                super().__init__(device=device, batch_size=[])
                self.conn         = connection
                self._last_action = torch.zeros(self.ACTION_DIM)
                self.last_reward  = 0.0

                lo = torch.tensor({lo_vals}, dtype=torch.float32)
                hi = torch.tensor({hi_vals}, dtype=torch.float32)

                self.observation_spec = CompositeSpec({{
                    "observation": BoundedTensorSpec(
                        low=lo, high=hi, shape=(self.STATE_DIM,), dtype=torch.float32, device=device
                    )
                }})
                self.action_spec = BoundedTensorSpec(
                    low=-1.0, high=1.0, shape=(self.ACTION_DIM,), dtype=torch.float32, device=device
                )
                self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,), dtype=torch.float32, device=device)

            def _reset(self, tensordict=None):
                self.conn.send([0.0] * self.ACTION_DIM)
                time.sleep(0.5)
                self._last_action = torch.zeros(self.ACTION_DIM)
                return TensorDict({{"observation": self._read_obs()}}, batch_size=[])

            def _step(self, tensordict):
                action = tensordict["action"].float().cpu()
                self.conn.send(action.tolist())
                next_obs         = self._read_obs()
                reward, done     = self._reward(action, next_obs)
                self.last_reward = reward
                self._last_action = action.clone()
                return TensorDict({{
                    "observation": next_obs,
                    "reward":     torch.tensor([reward], dtype=torch.float32),
                    "done":       torch.tensor([done],   dtype=torch.bool),
                    "terminated": torch.tensor([done],   dtype=torch.bool),
                    "truncated":  torch.tensor([False],  dtype=torch.bool),
                }}, batch_size=[])

            def _set_seed(self, seed):
                if seed is not None:
                    torch.manual_seed(seed)

            def _read_obs(self):
                vals = self.conn.receive(self.STATE_DIM)
                obs  = torch.tensor(vals, dtype=torch.float32)
                lo   = self.observation_spec["observation"].space.low
                hi   = self.observation_spec["observation"].space.high
                return obs.clamp(lo, hi)

            def _reward(self, action, next_obs):
                dist      = next_obs[{dist_idx}].item()
                collision = dist < self._COLLISION_DIST_CM
                speed     = action.abs().mean().item()
                jerk      = (action - self._last_action).abs().mean().item()

                reward = (
                    self._SPEED_WEIGHT      * speed
                    - self._COLLISION_PENALTY * float(collision)
                    - self._JERK_WEIGHT       * jerk
                )
                return float(reward), bool(collision)

            def close(self):
                self.conn.close()
    """)


# ── Training scripts ─────────────────────────────────────────────────────────

def _train(algorithm, priorities, motors, sensors):
    state_dim  = len(motors) + 2 + len(sensors)
    action_dim = len(motors)
    p          = priorities
    smap       = SLIDER_MAP[algorithm]

    if algorithm == 'random':
        return _train_random(action_dim)
    if algorithm == 'SAC':
        return _train_sac(p, smap, state_dim, action_dim, dropout=False)
    if algorithm == 'DroQ':
        return _train_sac(p, smap, state_dim, action_dim, dropout=True)
    if algorithm == 'TD3':
        return _train_td3(p, smap, state_dim, action_dim)
    return _train_sac(p, smap, state_dim, action_dim, dropout=False)


def _train_sac(p, smap, state_dim, action_dim, dropout):
    curiosity  = p.get('curiosity', 0.3)
    comfort    = p.get('comfort',   0.5)

    # Apply algorithm-specific slider formulas
    alpha_init = smap['curiosity']['formula'](curiosity)

    if dropout:  # DroQ — comfort controls both eps AND dropout
        soft_eps, dropout_p = smap['comfort']['formula'](comfort)
        variant = 'DroQ'
        utd     = 20
    else:        # SAC — comfort controls only eps
        soft_eps  = smap['comfort']['formula'](comfort)
        dropout_p = 0.0
        variant   = 'SAC'
        utd       = 1

    dropout_line = f"nn.Dropout(p={dropout_p:.4f})," if dropout else ""

    return dedent(f"""\
        # Generated by Kobe — {variant} training script
        # DO NOT EDIT — regenerate from Kobe IDE.
        #
        # Slider → TorchRL parameter mapping ({variant}):
        #
        #   curiosity  = {curiosity}
        #     └─ SACLoss(alpha_init={alpha_init:.4f})
        #        Higher curiosity → higher entropy temp → more random exploration
        #
        #   comfort    = {comfort}
        #     └─ SoftUpdate(eps={soft_eps:.4f})  ← target network smoothing
        #        {"└─ Q-network dropout(p=" + str(round(dropout_p, 4)) + ")  ← Q-value stability" if dropout else ""}
        #        Higher comfort → slower, smoother target updates
        #
        #   safety + efficiency → reward weights in environment.py

        import torch
        import torch.nn as nn
        from tensordict.nn import TensorDictModule
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal
        from torchrl.objectives import SACLoss
        from torchrl.objectives.utils import SoftUpdate
        from torchrl.data import ReplayBuffer, LazyMemmapStorage
        from environment import KobeEnv, HardwareConnection

        torch.manual_seed(42)

        STATE_DIM  = {state_dim}
        ACTION_DIM = {action_dim}

        print("Kobe: connecting to hardware...")
        conn = HardwareConnection(port='/dev/ttyACM0')
        env  = KobeEnv(connection=conn)
        print("Kobe: connected.")

        # ── Actor ─────────────────────────────────────────────────────────────
        # Outputs mean + log_std of TanhNormal.
        # alpha_init (curiosity slider) controls how stochastic this is.

        actor_net = MLP(
            in_features=STATE_DIM, out_features=2 * ACTION_DIM,
            num_cells=[256, 256], activation_class=nn.ReLU,
        )
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=actor_module, in_keys=["loc", "scale"], out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={{"low": -1.0, "high": 1.0}},
            return_log_prob=True,
        )

        # ── Q-networks ────────────────────────────────────────────────────────
        {"# DroQ: dropout(p=" + str(round(dropout_p, 4)) + ") from comfort slider = " + str(comfort) if dropout else "# SAC: standard Q-networks"}

        def _make_qnet():
            return TensorDictModule(
                nn.Sequential(
                    nn.Linear(STATE_DIM + ACTION_DIM, 256), nn.ReLU(),
                    {dropout_line}
                    nn.Linear(256, 256), nn.ReLU(),
                    {dropout_line}
                    nn.Linear(256, 1),
                ),
                in_keys=["observation", "action"], out_keys=["state_action_value"],
            )

        qnet = _make_qnet()

        # ── SAC loss ──────────────────────────────────────────────────────────
        # alpha_init = {alpha_init:.4f}  ← curiosity slider = {curiosity}

        loss_module = SACLoss(
            actor_network   = actor,
            qvalue_network  = qnet,
            num_qvalue_nets = 2,
            target_entropy  = "auto",
            alpha_init      = {alpha_init:.4f},   # ← curiosity = {curiosity}
            fixed_alpha     = False,
            gamma           = 0.99,
        )
        loss_module.make_value_estimator()

        # ── Target network ────────────────────────────────────────────────────
        # eps = {soft_eps:.4f}  ← comfort slider = {comfort}
        # Higher comfort → eps closer to 1.0 → smoother, slower target updates

        target_updater = SoftUpdate(loss_module, eps={soft_eps:.4f})   # ← comfort = {comfort}

        actor_optim  = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_optim = torch.optim.Adam(
            loss_module.qvalue_network_params.values(True, True), lr=3e-4
        )
        alpha_optim  = torch.optim.Adam([loss_module.log_alpha], lr=3e-4)

        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=100_000), batch_size=256
        )

        # ── Prefill ───────────────────────────────────────────────────────────

        print("Kobe: warming up replay buffer (10 random episodes)...")
        for ep in range(10):
            td = env.reset(); done = False
            while not done:
                td["action"] = env.action_spec.rand()
                td   = env.step(td)
                replay_buffer.add(td.clone())
                done = td["done"].item()
                td   = td.select("next").rename_key_("next", "")

        # ── Training ──────────────────────────────────────────────────────────

        UTD      = {utd}   {"# DroQ: high update-to-data ratio" if dropout else ""}
        EPISODES = {"100" if dropout else "300"}
        best_r   = float("-inf")

        print(f"Kobe: training {{EPISODES}} episodes ({variant})...")

        try:
            for ep in range(EPISODES):
                td = env.reset(); done = False; total_r = 0.0; steps = 0

                while not done and steps < 200:
                    with torch.no_grad():
                        td = actor(td)
                    td      = env.step(td)
                    total_r += env.last_reward
                    replay_buffer.add(td.clone())

                    for _ in range(UTD):
                        if len(replay_buffer) < 256: break
                        batch   = replay_buffer.sample()
                        loss_td = loss_module(batch)

                        actor_optim.zero_grad();  loss_td["loss_actor"].backward();  actor_optim.step()
                        critic_optim.zero_grad(); loss_td["loss_qvalue"].backward(); critic_optim.step()
                        alpha_optim.zero_grad();  loss_td["loss_alpha"].backward();  alpha_optim.step()
                        target_updater.step()

                    done  = td["done"].item()
                    td    = td.select("next").rename_key_("next", "")
                    steps += 1

                if total_r > best_r:
                    best_r = total_r
                    torch.save(actor.state_dict(), "kobe_policy_best.pt")

                print(f"Ep {{ep+1:>3}}/{{EPISODES}} | steps: {{steps:>3}} | "
                      f"reward: {{total_r:>9.2f}} | alpha: {{loss_module.alpha.item():.4f}} | best: {{best_r:>9.2f}}")

        except KeyboardInterrupt:
            print("\\nKobe: stopped early.")

        torch.save(actor.state_dict(), "kobe_policy_final.pt")
        print("Kobe: saved kobe_policy_final.pt and kobe_policy_best.pt")
        env.close()
    """)


def _train_td3(p, smap, state_dim, action_dim):
    curiosity = p.get('curiosity', 0.3)
    comfort   = p.get('comfort',   0.5)

    # TD3-specific slider formulas
    sigma_init          = smap['curiosity']['formula'](curiosity)
    soft_eps, noise_clip = smap['comfort']['formula'](comfort)

    return dedent(f"""\
        # Generated by Kobe — TD3 training script
        # DO NOT EDIT — regenerate from Kobe IDE.
        #
        # Slider → TorchRL parameter mapping (TD3):
        #
        #   curiosity  = {curiosity}
        #     └─ AdditiveGaussianWrapper(sigma_init={sigma_init:.4f})
        #        Higher curiosity → more exploration noise → bolder moves
        #
        #   comfort    = {comfort}
        #     └─ SoftUpdate(eps={soft_eps:.4f})       ← target smoothing
        #        └─ TD3Loss(noise_clip={noise_clip:.4f})  ← action precision
        #        Higher comfort → smoother target updates + tighter noise clipping
        #
        #   safety + efficiency → reward weights in environment.py

        import torch
        import torch.nn as nn
        from tensordict.nn import TensorDictModule
        from torchrl.modules import MLP, AdditiveGaussianWrapper
        from torchrl.objectives import TD3Loss
        from torchrl.objectives.utils import SoftUpdate
        from torchrl.data import ReplayBuffer, LazyMemmapStorage
        from environment import KobeEnv, HardwareConnection

        torch.manual_seed(42)

        STATE_DIM  = {state_dim}
        ACTION_DIM = {action_dim}

        print("Kobe: connecting to hardware...")
        conn = HardwareConnection(port='/dev/ttyACM0')
        env  = KobeEnv(connection=conn)
        print("Kobe: connected.")

        # ── Deterministic actor ───────────────────────────────────────────────

        actor_net = nn.Sequential(
            MLP(in_features=STATE_DIM, out_features=ACTION_DIM,
                num_cells=[256, 256], activation_class=nn.ReLU,
                activate_last_layer=False),
            nn.Tanh(),
        )
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["action"]
        )

        # Gaussian exploration noise.
        # sigma_init = {sigma_init:.4f}  ← curiosity slider = {curiosity}
        # Higher curiosity → more noise → robot tries more varied actions

        actor_explore = AdditiveGaussianWrapper(
            actor_module,
            sigma_init         = {sigma_init:.4f},   # ← curiosity = {curiosity}
            sigma_end          = {max(0.01, sigma_init * 0.1):.4f},
            annealing_num_steps = 30000,
            spec               = env.action_spec,
        )

        # ── Q-networks ────────────────────────────────────────────────────────

        def _make_qnet():
            return TensorDictModule(
                MLP(in_features=STATE_DIM + ACTION_DIM, out_features=1,
                    num_cells=[256, 256], activation_class=nn.ReLU),
                in_keys=["observation", "action"], out_keys=["state_action_value"],
            )

        qnet = _make_qnet()

        # ── TD3 loss ──────────────────────────────────────────────────────────
        # noise_clip = {noise_clip:.4f}  ← comfort slider = {comfort}
        # Higher comfort → tighter noise_clip → smoother, more deliberate actions

        loss_module = TD3Loss(
            actor_network   = actor_module,
            qvalue_network  = qnet,
            action_spec     = env.action_spec,
            policy_noise    = 0.2,
            noise_clip      = {noise_clip:.4f},   # ← comfort = {comfort}
            gamma           = 0.99,
            loss_function   = "l2",
            delay_actor     = 2,
        )
        loss_module.make_value_estimator()

        # Target network smoothing.
        # eps = {soft_eps:.4f}  ← comfort slider = {comfort}

        target_updater = SoftUpdate(loss_module, eps={soft_eps:.4f})   # ← comfort = {comfort}

        actor_optim  = torch.optim.Adam(actor_module.parameters(), lr=3e-4)
        critic_optim = torch.optim.Adam(
            loss_module.qvalue_network_params.values(True, True), lr=3e-4
        )

        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(max_size=100_000), batch_size=256
        )

        # ── Prefill ───────────────────────────────────────────────────────────

        print("Kobe: warming up (10 random episodes)...")
        for ep in range(10):
            td = env.reset(); done = False
            while not done:
                td["action"] = env.action_spec.rand()
                td   = env.step(td)
                replay_buffer.add(td.clone())
                done = td["done"].item()
                td   = td.select("next").rename_key_("next", "")

        # ── Training ──────────────────────────────────────────────────────────

        EPISODES  = 300
        best_r    = float("-inf")
        update_ctr = 0

        print(f"Kobe: training {{EPISODES}} episodes (TD3)...")

        try:
            for ep in range(EPISODES):
                td = env.reset(); done = False; total_r = 0.0; steps = 0

                while not done and steps < 200:
                    with torch.no_grad():
                        td = actor_explore(td)
                    td      = env.step(td)
                    total_r += env.last_reward
                    replay_buffer.add(td.clone())

                    if len(replay_buffer) >= 256:
                        batch   = replay_buffer.sample()
                        loss_td = loss_module(batch)
                        update_ctr += 1

                        critic_optim.zero_grad()
                        loss_td["loss_qvalue"].backward()
                        critic_optim.step()

                        if update_ctr % 2 == 0:   # TD3 delayed actor update
                            actor_optim.zero_grad()
                            loss_td["loss_actor"].backward()
                            actor_optim.step()
                            target_updater.step()

                    done  = td["done"].item()
                    td    = td.select("next").rename_key_("next", "")
                    steps += 1

                if total_r > best_r:
                    best_r = total_r
                    torch.save(actor_module.state_dict(), "kobe_policy_best.pt")

                print(f"Ep {{ep+1:>3}}/{{EPISODES}} | steps: {{steps:>3}} | "
                      f"reward: {{total_r:>9.2f}} | best: {{best_r:>9.2f}}")

        except KeyboardInterrupt:
            print("\\nKobe: stopped early.")

        torch.save(actor_module.state_dict(), "kobe_policy_final.pt")
        print("Kobe: saved kobe_policy_final.pt and kobe_policy_best.pt")
        env.close()
    """)


def _train_random(action_dim):
    return dedent(f"""\
        # Generated by Kobe — Random Policy (hardware test, no learning)
        import torch
        from environment import KobeEnv, HardwareConnection

        conn = HardwareConnection(port='/dev/ttyACM0')
        env  = KobeEnv(connection=conn)
        print("Kobe: running random policy — hardware test only")

        for ep in range(20):
            td = env.reset(); done = False; steps = 0
            while not done and steps < 100:
                td["action"] = env.action_spec.rand()
                td    = env.step(td)
                done  = td["done"].item()
                td    = td.select("next").rename_key_("next", "")
                steps += 1
            print(f"Episode {{ep+1}}/20 — {{steps}} steps")

        env.close()
    """)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_safety_threshold(ir):
    for instr in ir:
        if instr.get('op') == 'CMP_DIST' and instr.get('comparator') in ('<', '<='):
            return float(instr.get('valueCm', 20.0))
    return 20.0

def _lo(s):
    t = s['type']
    return EV3_SENSOR[t]['lo'] if t in EV3_SENSOR else RPI_SENSOR.get(t, {}).get('lo', 0.0)

def _hi(s):
    t = s['type']
    return EV3_SENSOR[t]['hi'] if t in EV3_SENSOR else RPI_SENSOR.get(t, {}).get('hi', 1.0)


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_ir = [
        {'op': 'POLICY', 'curiosity': 0.7, 'safety': 0.9, 'comfort': 0.3, 'efficiency': 0.6},
        {'op': 'CMP_DIST', 'comparator': '<', 'valueCm': 20.0},
        {'op': 'HALT'},
    ]
    test_p  = {'curiosity': 0.7, 'safety': 0.9, 'comfort': 0.3, 'efficiency': 0.6}
    test_hw = {'target': 'EV3',
               'motors':  [{'port': 'A'}, {'port': 'B'}],
               'sensors': [{'type': 'dist', 'port': '1'}]}

    for alg in ('SAC', 'TD3', 'DroQ'):
        print(f"\n{'='*60}  {alg} slider descriptions  {'='*60}")
        for k, v in slider_descriptions(alg).items():
            print(f"  {k:10} → {v['param']}")
            print(f"             {v['effect']}")

        files = generate(test_ir, test_p, test_hw, alg)
        print(f"\n  Generated: {list(files.keys())}")