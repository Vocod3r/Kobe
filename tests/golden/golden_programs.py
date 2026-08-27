"""
Golden programs for semantic equivalence testing.

Each golden program is a minimal, well-formed Kobe program that tests a specific
compiler feature. Trace equivalence (reference interpreter vs IR executor) confirms
both the compiler and IR executor handle that feature correctly.
"""

# Test 1: Simple walk-stop sequence
GOLDEN_SIMPLE_WALK = """
policy {
  safety = 0.8;
}

walk forward;
stop;
"""

# Test 2: Loop for with fixed count
GOLDEN_LOOP_FOR = """
policy {
  safety = 0.8;
}

loop (3) {
  walk forward;
}
stop;
"""

# Test 3: Observe with simple condition
GOLDEN_OBSERVE = """
hardware {
  sensors: [dist@1]
}

policy {
  safety = 0.8;
}

observe(dist) {
  dist < 50 cm then {
    walk forward;
  }
}
"""

# Test 4: Basic if-then-else
GOLDEN_IF_THEN_ELSE = """
hardware {
  sensors: [dist@1]
}

policy {
  safety = 0.8;
}

observe(dist) {
  dist < 30 cm then {
    stop;
  }
  else {
    walk forward;
  }
}
"""

# Test 5: Turn action
GOLDEN_TURN = """
policy {
  safety = 0.8;
}

turn left;
walk forward;
stop;
"""

# Test 6: Multiple actions
GOLDEN_MULTIPLE_ACTIONS = """
policy {
  safety = 0.8;
}

walk forward;
turn left;
walk forward;
turn right;
stop;
"""

# Test 7: Wait action
GOLDEN_WAIT = """
policy {
  safety = 0.8;
}

walk forward;
wait 1 sec;
stop;
"""

# Test 8: Nested loop
GOLDEN_NESTED_LOOP = """
policy {
  safety = 0.8;
}

loop (2) {
  loop (2) {
    walk forward;
  }
  turn left;
}
stop;
"""

# Test 9: Loop with break
GOLDEN_LOOP_WITH_BREAK = """
hardware {
  sensors: [dist@1]
}

policy {
  safety = 0.8;
}

loop (5) {
  observe(dist) {
    dist < 20 cm then {
      break;
    }
  }
  walk forward;
}
stop;
"""

# Test 10: Colour sensor condition
GOLDEN_COLOUR_CONDITION = """
hardware {
  sensors: [colour@1]
}

policy {
  safety = 0.8;
}

observe(colour) {
  colour is red then {
    stop;
  }
  else {
    walk forward;
  }
}
"""

GOLDEN_PROGRAMS = {
    'simple_walk': GOLDEN_SIMPLE_WALK,
    'loop_for': GOLDEN_LOOP_FOR,
    'observe': GOLDEN_OBSERVE,
    'if_then_else': GOLDEN_IF_THEN_ELSE,
    'turn': GOLDEN_TURN,
    'multiple_actions': GOLDEN_MULTIPLE_ACTIONS,
    'wait': GOLDEN_WAIT,
    'nested_loop': GOLDEN_NESTED_LOOP,
    'loop_with_break': GOLDEN_LOOP_WITH_BREAK,
    'colour_condition': GOLDEN_COLOUR_CONDITION,
}
