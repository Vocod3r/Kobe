#!/usr/bin/env python3
"""stdin/stdout JSON bridge: compile Kobe source for the Electron IDE."""
from __future__ import annotations

import json
import sys
import traceback

# Allow imports from project root
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

from pipeline import run


def main() -> int:
    try:
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        source = req.get('source', '')
        trial_level = req.get('trialLevel', 2)
        target = req.get('target', 'rl')
        result = run(source, trial_level=trial_level, target=target)
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        return 0
    except Exception as exc:
        err = {'error': str(exc), 'traceback': traceback.format_exc()}
        sys.stdout.write(json.dumps(err))
        sys.stdout.flush()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
