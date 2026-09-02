import random
from typing import Tuple, Dict, Any, List

SENSORS = ['dist', 'colour', 'touch', 'IR', 'UV', 'gyro', 'sound']
COMPARATORS = ['<=', '>=', '==', '!=', '<', '>']
COLOURS = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white', 'black', 'none']
DIRECTIONS = ['forward', 'backward', 'left', 'right']
SPEEDS = ['slowly', 'normally', 'quickly']
UNITS = {'dist': ['cm', 'm', 'in'], 'wait': ['ms', 'sec']}


class OrderedSet:
    """Deterministic set that preserves insertion order."""
    def __init__(self, items=None):
        self._items = dict.fromkeys(items or [])

    def add(self, item):
        self._items[item] = None

    def update(self, items):
        for item in items:
            self._items[item] = None

    def remove(self, item):
        self._items.pop(item, None)

    def discard(self, item):
        self._items.pop(item, None)

    def clear(self):
        self._items.clear()

    def __iter__(self):
        return iter(self._items.keys())

    def __len__(self):
        return len(self._items)

    def __contains__(self, item):
        return item in self._items

    def __bool__(self):
        return bool(self._items)

    def to_list(self) -> List[Any]:
        return list(self._items.keys())


class ProgramGenerator:
    def __init__(self, seed: int, max_depth: int = 3, max_statements: int = 10, loop_until_sat_prob: float = 0.7):
        self.rng = random.Random(seed)
        self.max_depth = max_depth
        self.max_statements = max_statements
        self.loop_until_sat_prob = loop_until_sat_prob
        self.sensors_used = OrderedSet()
        self.comparators_used = OrderedSet()
        self.loop_forms_used = OrderedSet()
        self.branch_forms_used = OrderedSet()
        self.max_depth_reached = 0

    def generate(self) -> Tuple[str, Dict[str, Any]]:
        self.sensors_used.clear()
        self.comparators_used.clear()
        self.loop_forms_used.clear()
        self.branch_forms_used.clear()
        self.max_depth_reached = 0

        # Generate body first so we know which sensors to declare
        body_statements = self._generate_block(depth=0)
        
        hardware_block = self._generate_hardware()
        
        source = f"algorithm DroQ\n\n{hardware_block}\n\npolicy {{ safety = 0.9; }}\n\n"
        source += "\n".join(body_statements)

        coverage = {
            'sensors_exercised': self.sensors_used.to_list(),
            'sensors_coverage_pct': len(self.sensors_used) / len(SENSORS) * 100,
            'comparators_exercised': self.comparators_used.to_list(),
            'comparators_coverage_pct': len(self.comparators_used) / len(COMPARATORS) * 100,
            'loop_forms_used': self.loop_forms_used.to_list(),
            'branch_forms_used': self.branch_forms_used.to_list(),
            'max_depth_reached': self.max_depth_reached
        }
        return source, coverage

    def _generate_hardware(self) -> str:
        sensors_decl = []
        for i, s in enumerate(self.sensors_used):
            sensors_decl.append(f"{s}@{i+1}")
        
        sensors_str = ""
        if sensors_decl:
            sensors_str = f"  sensors: [{', '.join(sensors_decl)}]\n"
        
        return f"hardware {{\n  target: EV3\n  motors: [A, B]\n{sensors_str}}}"

    def _generate_block(self, depth: int) -> List[str]:
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        num_stmts = self.rng.randint(1, max(1, self.max_statements // max(1, depth)))
        statements = []
        for _ in range(num_stmts):
            statements.append(self._generate_statement(depth))
        return statements

    def _generate_statement(self, depth: int) -> str:
        return self._generate_stmt(depth, in_loop=False)

    def _generate_stmt(self, depth: int, in_loop: bool) -> str:
        choices = ['walk', 'run', 'turn', 'stop', 'wait']
        
        if depth < self.max_depth:
            choices.extend(['observe', 'if', 'loop_for', 'loop_until'])
        if in_loop:
            choices.append('break')
            
        weights = [1] * len(choices)
        choice = self.rng.choices(choices, weights=weights)[0]
        
        if choice == 'walk':
            return f"walk {self.rng.choice(DIRECTIONS)} {self.rng.choice(SPEEDS)};"
        elif choice == 'run':
            return f"run {self.rng.choice(DIRECTIONS)} {self.rng.choice(SPEEDS)};"
        elif choice == 'turn':
            return f"turn {self.rng.choice(['left', 'right'])};"
        elif choice == 'stop':
            return "stop;"
        elif choice == 'wait':
            return f"wait {self.rng.randint(1, 100)} {self.rng.choice(UNITS['wait'])};"
        elif choice == 'break':
            return "break;"
        elif choice == 'observe':
            self.branch_forms_used.add('observe')
            sensors = self.rng.sample(SENSORS, k=self.rng.randint(1, min(3, len(SENSORS))))
            self.sensors_used.update(sensors)
            
            branches = []
            for _ in range(self.rng.randint(1, 3)):
                cond = self._generate_condition(sensors)
                then_block = self._generate_block_str(depth + 1, in_loop)
                branches.append(f"{cond} then {then_block}")
                if self.rng.random() < 0.3:
                    else_block = self._generate_block_str(depth + 1, in_loop)
                    branches[-1] += f" else {else_block}"
            
            sensors_str = ", ".join(sensors)
            branches_str = "\n".join(branches)
            return f"observe({sensors_str}) {{\n{branches_str}\n}}"
            
        elif choice == 'if':
            self.branch_forms_used.add('if')
            cond = self._generate_condition()
            then_block = self._generate_block_str(depth + 1, in_loop)
            stmt = f"if {cond} then {then_block}"
            if self.rng.random() < 0.5:
                stmt += f" else {self._generate_block_str(depth + 1, in_loop)}"
            return stmt
            
        elif choice == 'loop_for':
            self.loop_forms_used.add('for')
            count = self.rng.randint(1, 5)
            body = self._generate_block_str(depth + 1, True)
            return f"loop ({count}) {body}"
            
        elif choice == 'loop_until':
            self.loop_forms_used.add('until')
            cond = self._generate_loop_until_condition()
            body = self._generate_block_str(depth + 1, True)
            return f"loop until ({cond}) {body}"
            
        return "stop;"

    def _generate_block_str(self, depth: int, in_loop: bool) -> str:
        stmts = []
        for _ in range(self.rng.randint(1, 3)):
            stmts.append(self._generate_stmt(depth, in_loop))
        stmts_str = " ".join(stmts)
        return f"{{ {stmts_str} }}"

    def _generate_condition(self, available_sensors=None) -> str:
        if available_sensors is None:
            sensor = self.rng.choice(self.sensors_used.to_list() if self.sensors_used else SENSORS)
        else:
            sensor = self.rng.choice(available_sensors)
            
        self.sensors_used.add(sensor)
        
        if self.rng.random() < 0.2:
            return self._generate_compound_condition(available_sensors)
            
        return self._generate_atomic_condition(sensor)

    def _generate_loop_until_condition(self, available_sensors=None) -> str:
        """Generate a condition for loop_until, biased towards satisfiability in the scenario."""
        if self.rng.random() < self.loop_until_sat_prob:
            return self._generate_satisfiable_condition(available_sensors)
        else:
            return self._generate_condition(available_sensors)

    def _generate_satisfiable_condition(self, available_sensors=None) -> str:
        """Generate a condition that is provably satisfiable against default_scenario sensor readings."""
        if available_sensors is None:
            sensor = self.rng.choice(self.sensors_used.to_list() if self.sensors_used else SENSORS)
        else:
            sensor = self.rng.choice(available_sensors)

        self.sensors_used.add(sensor)

        if self.rng.random() < 0.15:
            op = self.rng.choice(['and', 'or', 'not'])
            if op == 'not':
                # not of an unsatisfiable or specific condition, or colour negation
                if sensor == 'colour':
                    return f"not (colour is {self.rng.choice(['yellow', 'black', 'white', 'orange'])})"
                elif sensor == 'dist':
                    return f"not (dist > 150 cm)"
                elif sensor == 'sound':
                    return f"not (sound > 150 db)"
                return f"not (gyro tilt > 150 deg)"
            elif op == 'or':
                left = self._generate_satisfiable_atomic(sensor)
                right = self._generate_atomic_condition()
                return f"({left} or {right})"
            else:
                left = self._generate_satisfiable_atomic(sensor)
                right = self._generate_satisfiable_atomic()
                return f"({left} and {right})"

        return self._generate_satisfiable_atomic(sensor)

    def _generate_satisfiable_atomic(self, sensor=None) -> str:
        if sensor is None:
            sensor = self.rng.choice(self.sensors_used.to_list() if self.sensors_used else SENSORS)
        elif isinstance(sensor, list):
            sensor = self.rng.choice(sensor)

        self.sensors_used.add(sensor)

        if sensor == 'dist':
            # Scenario distance readings: 100, 90, 80, 70, 60, 50, 40, 30, 25, 20, 15 cm
            cmp = self.rng.choice(COMPARATORS)
            self.comparators_used.add(cmp)
            if cmp == '==':
                val = self.rng.choice([15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100])
            elif cmp in ['<=', '<']:
                val = self.rng.choice([25, 30, 40, 50, 60, 70, 80, 90, 100])
            elif cmp in ['>=', '>']:
                val = self.rng.choice([15, 20, 25, 30, 40, 50, 60, 70])
            else:  # '!='
                val = self.rng.choice([0, 5, 120])
            return f"dist {cmp} {val} cm"

        elif sensor == 'colour':
            # Scenario colour readings: red, green, blue, none
            if self.rng.random() < 0.6:
                return f"colour is {self.rng.choice(['red', 'green', 'blue', 'none'])}"
            else:
                return f"colour not {self.rng.choice(['yellow', 'black', 'white', 'orange', 'indigo', 'violet'])}"

        elif sensor == 'touch':
            # Scenario touch readings: False, False, False, True
            return "touch pressed"

        elif sensor == 'IR':
            # Scenario IR readings: False, False, True, False
            if self.rng.random() < 0.7:
                return "IR detected"
            else:
                cmp = self.rng.choice(['<=', '>=', '==', '!='])
                self.comparators_used.add(cmp)
                return f"IR signal {cmp} {self.rng.choice([0, 1])}"

        elif sensor == 'UV':
            # Scenario UV readings: 0.0, 2.0, 5.0, 8.0
            if self.rng.random() < 0.5:
                return "UV detected"
            else:
                cmp = self.rng.choice(['<=', '>=', '<', '>', '=='])
                self.comparators_used.add(cmp)
                return f"UV index {cmp} {self.rng.choice([0, 2, 5, 8])}"

        elif sensor == 'gyro':
            # Scenario gyro readings: 0.0, 15.0, -15.0, 30.0
            cmp = self.rng.choice(COMPARATORS)
            self.comparators_used.add(cmp)
            if cmp in ['<=', '<']:
                val = self.rng.choice([15, 20, 30, 45, 90])
            elif cmp in ['>=', '>']:
                val = self.rng.choice([0, 10, 15, 25])
            elif cmp == '==':
                val = self.rng.choice([0, 15, 30])
            else:
                val = self.rng.choice([45, 90, 180])
            return f"gyro tilt {cmp} {val} deg"

        elif sensor == 'sound':
            # Scenario sound readings: 30.0, 50.0, 70.0, 90.0
            cmp = self.rng.choice(COMPARATORS)
            self.comparators_used.add(cmp)
            if cmp in ['<=', '<']:
                val = self.rng.choice([40, 50, 60, 70, 80, 90, 100])
            elif cmp in ['>=', '>']:
                val = self.rng.choice([20, 30, 40, 50, 60, 70])
            elif cmp == '==':
                val = self.rng.choice([30, 50, 70, 90])
            else:
                val = self.rng.choice([10, 20, 110])
            return f"sound {cmp} {val} db"

        return "touch pressed"

    def _generate_compound_condition(self, available_sensors) -> str:
        op = self.rng.choice(['and', 'or', 'not'])
        if op == 'not':
            return f"not ({self._generate_atomic_condition(available_sensors)})"
        else:
            left = self._generate_atomic_condition(available_sensors)
            right = self._generate_atomic_condition(available_sensors)
            return f"({left} {op} {right})"

    def _generate_atomic_condition(self, sensor=None) -> str:
        if sensor is None:
            sensor = self.rng.choice(self.sensors_used.to_list() if self.sensors_used else SENSORS)
        elif isinstance(sensor, list):
            sensor = self.rng.choice(sensor)
        
        self.sensors_used.add(sensor)
        cmp = self.rng.choice(COMPARATORS)
        self.comparators_used.add(cmp)
        
        if sensor == 'dist':
            return f"dist {cmp} {self.rng.randint(5, 100)} {self.rng.choice(UNITS['dist'])}"
        elif sensor == 'colour':
            self.comparators_used.remove(cmp)
            if self.rng.random() < 0.5:
                return f"colour is {self.rng.choice(COLOURS)}"
            else:
                return f"colour not {self.rng.choice(COLOURS)}"
        elif sensor == 'touch':
            self.comparators_used.remove(cmp)
            return "touch pressed"
        elif sensor == 'IR':
            if self.rng.random() < 0.5:
                self.comparators_used.remove(cmp)
                return "IR detected"
            return f"IR signal {cmp} {self.rng.randint(0, 100)}"
        elif sensor == 'UV':
            if self.rng.random() < 0.5:
                self.comparators_used.remove(cmp)
                return "UV detected"
            return f"UV index {cmp} {self.rng.randint(0, 11)}"
        elif sensor == 'gyro':
            return f"gyro tilt {cmp} {self.rng.randint(0, 180)} deg"
        elif sensor == 'sound':
            return f"sound {cmp} {self.rng.randint(0, 190)} db"

        return "touch pressed"


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from parser import parse


def generate_corpus(n: int, seed: int, max_depth: int = 3, loop_until_sat_prob: float = 0.7) -> List[Tuple[str, Dict[str, Any]]]:
    gen = ProgramGenerator(seed, max_depth=max_depth, loop_until_sat_prob=loop_until_sat_prob)
    corpus = []
    for _ in range(n):
        src, cov = gen.generate()
        try:
            parse(src)
        except Exception as e:
            print(f"GENERATOR BUG: Produced invalid syntax!\nError: {e}\nSource:\n{src}", file=sys.stderr)
            raise AssertionError("Generator produced invalid syntax") from e
        corpus.append((src, cov))
    return corpus


if __name__ == "__main__":
    c = generate_corpus(2, 42)
    for src, cov in c:
        print(src)
        print(cov)
        print("---")
