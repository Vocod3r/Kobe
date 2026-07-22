# priorities.py

DEFAULTS = {
    'curiosity':  0.3,
    'safety':     0.5,
    'comfort':    0.5,
    'efficiency': 0.5
}

def extract_priorities(ast: dict) -> dict:
    """
    Returns merged priorities:
    - Explicit policy{} values take precedence
    - Implicit signals fill in anything not explicitly set
    """
    explicit = _get_explicit(ast['policy'])
    implicit = _get_implicit(ast['body'])

    return {
        key: explicit[key] if key in explicit else implicit.get(key, DEFAULTS[key])
        for key in ('curiosity', 'safety', 'comfort', 'efficiency')
    }

def _get_explicit(policy: dict | None) -> dict:
    if policy is None:
        return {}
    result = {}
    for key in ('curiosity', 'safety', 'comfort', 'efficiency'):
        val = policy.get(key)
        if val is not None and val != DEFAULTS.get(key):
            result[key] = val
    return result

def _get_implicit(body: list) -> dict:
    scores = {'efficiency': 0, 'safety': 0, 'comfort': 0}
    _score_statements(body, scores)

    total = sum(scores.values())
    if total == 0:
        return DEFAULTS.copy()

    # Normalize to [0, 1]
    return {k: round(v / total, 3) for k, v in scores.items()}

def _score_statements(stmts: list, scores: dict):
    for stmt in stmts:
        _score_node(stmt, scores)

def _score_node(node: dict, scores: dict):
    t = node['type']

    if t == 'Walk':
        speed = node.get('speed', 'normally')
        if speed == 'slowly':
            scores['efficiency'] -= 1
            scores['safety']     += 2
            scores['comfort']    += 2
        elif speed == 'quickly':
            scores['efficiency'] += 2
            scores['safety']     -= 1
            scores['comfort']    -= 1
        else:
            scores['efficiency'] += 1
            scores['safety']     += 1
            scores['comfort']    += 1

    elif t == 'Run':
        speed = node.get('speed', 'normally')
        if speed == 'quickly':
            scores['efficiency'] += 4
            scores['safety']     -= 2
            scores['comfort']    -= 1
        else:
            scores['efficiency'] += 3
            scores['safety']     -= 1

    elif t == 'Stop':
        scores['efficiency'] -= 1
        scores['safety']     += 3

    elif t == 'Wait':
        scores['efficiency'] -= 1
        scores['safety']     += 1
        scores['comfort']    += 1

    elif t == 'Observe':
        for sensor in set(node['sensors']):
            if sensor in ('dist', 'IR', 'UV', 'touch'):
                scores['safety']  += 2
            elif sensor == 'colour':
                scores['safety']  += 1
                scores['comfort'] += 1
            elif sensor == 'gyro':
                scores['safety']  += 1
                scores['comfort'] += 2

        for branch in node['branches']:
            _score_condition(branch['condition'], scores)
            _score_statements(branch['then'],  scores)
            _score_statements(branch['else'],  scores)

    elif t == 'If':
        _score_condition(node['condition'], scores)
        _score_statements(node['then'], scores)
        _score_statements(node['else'], scores)

    elif t in ('LoopFor', 'LoopUntil'):
        scores['efficiency'] += 1
        _score_statements(node['body'], scores)

    elif t == 'Break':
        scores['safety']  += 1
        scores['comfort'] += 1

def _score_condition(cond: dict, scores: dict):
    t = cond['type']

    if t in ('And', 'Or'):
        _score_condition(cond['left'],  scores)
        _score_condition(cond['right'], scores)

    elif t == 'Not':
        _score_condition(cond['operand'], scores)

    elif t == 'DistCondition':
        val = cond['value']
        # Convert to cm for comparison
        cm = _to_cm(val, cond['unit'])
        if cm < 20:
            scores['safety'] += 3
        elif cm > 50:
            scores['efficiency'] += 2

    # Stop inside dist branch is caught at statement level — no extra scoring needed

def _to_cm(value: float, unit: str) -> float:
    if unit == 'm':  return value * 100
    if unit == 'in': return value * 2.54
    return value