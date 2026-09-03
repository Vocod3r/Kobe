# Kobe — Robotics RL Language

Kobe is a domain-specific language for teaching children to program reinforcement-learning robots.

## Project structure

```
Kobe/
├── lexer.py, parser.py, semantic_analyzer.py   # Front-end
├── priorities.py, compiler.py                  # IR compilation
├── codegen.py, backend.py                      # Code generation + live training
├── pipeline.py                                 # End-to-end compile entry point
├── ipc/                                        # Electron IPC bridges
│   ├── pipeline_ipc.py
│   └── train_ipc.py
└── ide/                                        # Electron + React IDE
```

## Setup

### Python (compiler + training)

```bash
pip install -r requirements.txt
```

### Electron IDE

```bash
cd ide
npm install
npm run dev
```

## Usage

1. Write a Kobe program in the editor pane.
2. Diagnostics, IR, and generated TorchRL training scripts appear in the inspector.
3. Adjust policy sliders — descriptions update per algorithm.
4. Click **Train Robot** to run live simulator training (powered by Kobe's built-in PyTorch actor-critic backend).

## Pipeline test (CLI)

```bash
python -c "from pipeline import run; import json; print(json.dumps(run(open('example.kobe').read()), indent=2))"
```
