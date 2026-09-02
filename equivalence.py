"""
Shared semantic-equivalence machinery: a default deterministic sensor scenario
and a trace comparator, used by both tests/golden/run_golden_tests.py and
gate2_validator.py so the two can't silently drift apart.
"""
from __future__ import annotations

from reference_interpreter import Scenario, ReferenceInterpreter, TraceEvent
from ir_trace_executor import IRTraceExecutor


def default_scenario() -> Scenario:
    """A fixed, deterministic sensor scenario covering all sensor kinds and a
    range of distance thresholds, so both DistCondition and colour/touch/IR/UV/
    gyro/sound branches get exercised."""
    return Scenario(
        distance_readings=[100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 15.0],
        colour_readings=[
            {'name': 'red', 'nm': 650},
            {'name': 'green', 'nm': 530},
            {'name': 'blue', 'nm': 470},
            {'name': 'none', 'nm': 600},
        ],
        ir_readings=[False, False, True, False],
        touch_readings=[False, False, False, True],
        uv_readings=[0.0, 2.0, 5.0, 8.0],
        gyro_readings=[0.0, 15.0, -15.0, 30.0],
        sound_readings=[30.0, 50.0, 70.0, 90.0],
        distance_change_per_step=0.0,  # fixed for determinism
    )


def compare_traces(
    ref_events: list[TraceEvent],
    ir_events: list[TraceEvent],
    ref_hit_cap: bool = False,
    ir_hit_cap: bool = False,
) -> tuple[bool, str, bool]:
    """Compare two observable-event traces for equivalence.

    Observable events: action, observe, branch, break, halt (loop_iter is
    informational only and already excluded by Trace.observable_events()).

    Returns:
        (passed, message, is_capped_equivalent)
    """
    both_capped = ref_hit_cap and ir_hit_cap

    if not both_capped:
        if len(ref_events) != len(ir_events):
            return False, f"Event count mismatch: ref={len(ref_events)}, ir={len(ir_events)}", False
        events_to_compare = len(ref_events)
    else:
        events_to_compare = min(len(ref_events), len(ir_events))

    for i in range(events_to_compare):
        ref_evt = ref_events[i]
        ir_evt = ir_events[i]

        if ref_evt.kind != ir_evt.kind:
            return False, f"Event {i} kind mismatch: ref={ref_evt.kind}, ir={ir_evt.kind}", False

        if ref_evt.kind == 'action':
            if ref_evt.details.get('action') != ir_evt.details.get('action'):
                return False, f"Event {i} action mismatch: ref={ref_evt.details.get('action')}, ir={ir_evt.details.get('action')}", False
            if ref_evt.details.get('direction') != ir_evt.details.get('direction'):
                return False, f"Event {i} action direction mismatch: ref={ref_evt.details.get('direction')}, ir={ir_evt.details.get('direction')}", False

        elif ref_evt.kind == 'observe':
            if ref_evt.details.get('sensors') != ir_evt.details.get('sensors'):
                return False, f"Event {i} observe sensors mismatch", False
            ref_readings = ref_evt.details.get('readings', {})
            ir_readings = ir_evt.details.get('readings', {})
            for sensor in ref_readings:
                if ref_readings.get(sensor) != ir_readings.get(sensor):
                    return False, (
                        f"Event {i} reading mismatch for {sensor}: "
                        f"ref={ref_readings.get(sensor)}, ir={ir_readings.get(sensor)}"
                    ), False

        elif ref_evt.kind == 'branch':
            if ref_evt.details.get('result') != ir_evt.details.get('result'):
                return False, (
                    f"Event {i} branch result mismatch: "
                    f"ref={ref_evt.details.get('result')}, ir={ir_evt.details.get('result')}"
                ), False

    if both_capped:
        return True, f"Capped-equivalent (shared prefix of {events_to_compare} events match up to cap)", True

    return True, "All events match", False


def check_equivalence(ast: dict, ir: list[dict], scenario: Scenario | None = None, max_steps: int | None = None) -> dict:
    """Run the reference interpreter (AST) and the IR trace executor (compiled
    IR) against the same scenario and report whether their observable traces
    match. This is the actual proof that compiler.py's flattened IR preserves
    source-level semantics — the core claim of Gate 2.

    max_steps: optional step cap for both executors (only used in fuzzing/testing
    to prevent infinite loops in randomly generated programs). Leave as None for
    production Gate 2 checks so real program behavior is never silently truncated.
    """
    scenario = scenario or default_scenario()
    out = {
        'passed': False,
        'capped_equivalent': False,
        'message': '',
        'ref_event_count': None,
        'ir_event_count': None,
        'ref_hit_cap': False,
        'ir_hit_cap': False,
    }

    try:
        ref_interp = ReferenceInterpreter(ast, scenario, max_steps=max_steps)
        ref_trace = ref_interp.execute()
        ref_events = ref_trace.observable_events()
        ref_hit_cap = getattr(ref_interp, 'hit_cap', False) or (max_steps is not None and ref_interp.step > max_steps)
    except Exception as e:
        out['message'] = f'Reference interpreter error: {e}'
        return out

    try:
        ir_exec = IRTraceExecutor(ir, scenario, max_steps=max_steps)
        ir_trace = ir_exec.execute()
        ir_events = ir_trace.observable_events()
        ir_hit_cap = getattr(ir_exec, 'hit_cap', False) or (max_steps is not None and ir_exec.step > max_steps)
    except Exception as e:
        out['message'] = f'IR executor error: {e}'
        return out

    out['ref_event_count'] = len(ref_events)
    out['ir_event_count'] = len(ir_events)
    out['ref_hit_cap'] = ref_hit_cap
    out['ir_hit_cap'] = ir_hit_cap

    passed, message, capped_equivalent = compare_traces(
        ref_events, ir_events, ref_hit_cap=ref_hit_cap, ir_hit_cap=ir_hit_cap
    )
    out['passed'] = passed
    out['capped_equivalent'] = capped_equivalent
    out['message'] = message
    return out