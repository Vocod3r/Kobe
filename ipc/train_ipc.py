#!/usr/bin/env python3
"""stdin/stdout JSON bridge: train a Kobe policy for the Electron IDE."""
from __future__ import annotations

import json
import logging
import sys
import traceback

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from backend import train_policy

# PyTorch logs to stderr so stdout stays clean JSON lines.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING)


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + '\n')
    sys.stdout.flush()


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        ir = req['ir']
        priorities = req['priorities']
        algorithm = req.get('algorithm', 'SAC')
        trial_level = req.get('trialLevel', 2)
        seed = req.get('seed')

        def on_progress(msg: dict) -> None:
            emit(msg)

        result = train_policy(
            ir=ir,
            priorities=priorities,
            algorithm=algorithm,
            trial_level=trial_level,
            progress_callback=on_progress,
            seed=seed,
        )
        emit({'type': 'done', **result})
        return 0
    except Exception as exc:
        emit({'type': 'error', 'error': str(exc), 'traceback': traceback.format_exc()})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
