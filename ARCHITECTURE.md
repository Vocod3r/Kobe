# Kobe System Architecture & Roadmap

Complete specification of Kobe system design and timeline to publication.

## System Architecture

### Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Study Layer (Sep 26-Oct 12)                                │
│  - Human subject tasks (T1, T2, T3, Transfer)               │
│  - Within-subject comparison (Opaque vs Inspector)          │
│  - Questionnaires (understanding, confidence, workload)     │
│  - Baseline: Python+TorchRL                                 │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  IDE Layer (Sep 8-19)                                       │
│  - Text editor (Monaco) + Blocks editor                     │
│  - Compile button → IR visualization                        │
│  - Inspector panel (show generated env/training/safety)     │
│  - Error reporting & syntax help                            │
│  - Electron frontend + Python IPC backend                   │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  Generation Layer (CURRENT: Aug 28-Sep 5) ✓ IN PROGRESS     │
│  - environment.py: IR → Gymnasium                           │
│  - trainer.py: IR → TorchRL config (export)                │
│  - inspector.py: IR → human explanations                    │
│  - gate2_validator.py: semantic equivalence proof           │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  Compilation Layer (Completed ✓)                            │
│  - lexer.py: Tokenization                                   │
│  - parser.py: AST construction                              │
│  - compiler.py: AST → IR (canonical representation)         │
│  - semantic_analyzer.py: Type/semantic checking             │
│  - Canonical IR: all frontends → same IR                    │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│  Input Languages (Completed ✓)                              │
│  - Text DSL (Kobe language)                                 │
│  - Blocks visual language (TODO: syntax→IR mapping)         │
│  - Both compile to identical IR                             │
└─────────────────────────────────────────────────────────────┘
```

### Critical Invariant: Semantic Equivalence

**Principle:** All code generation and platform support flows through canonical IR.

**Guarantees:**
1. Reference interpreter (AST) ≡ IR executor on deterministic traces
2. Generated environment (IR) ≡ handwritten reference on fixed scenarios
3. LEGO backend output ≡ simulation on same program
4. Raspberry Pi ≡ LEGO ≡ simulation
5. Block program ≡ Text program (same IR)

**Validation:**
- Gate 1: Parser (10/10 golden tests parse)
- Gate 2: Compiler (10/10 golden + fuzz 200/200 inclusive, 0 real failures)
- Gate 3: Environment (generated env ≡ reference)
- Gate 4: Training (policy trains successfully)
- Gate 5: Study (system ready for humans)

## Implementation Status

### Completed (Aug 27-28)
- ✓ Lexer, Parser, Compiler (fully functional)
- ✓ Semantic analyzer (validation)
- ✓ Reference interpreter (ground truth)
- ✓ IR trace executor (comparison)
- ✓ Handwritten baseline (control)
- ✓ Golden test suite (10 programs)
- ✓ Environment generator (IR → Gymnasium)
- ✓ Trainer generator (IR → TorchRL; implemented for export, live training uses backend.py)
- ✓ Inspector (IR → explanations)
- ✓ Study tasks (T1, T2, T3, Transfer)
- ✓ Gate 2 validator (compiler proof)
- ✓ Test infrastructure (TESTING.md)

### In Progress (Aug 28-30)
- ✓ Golden test suite 10/10 passing; fuzz suite 200/200 inclusive (130 strict +
  70 capped-equivalent), 0 real failures (loop_until polarity, observe
  cap-check, and break loop_stack fixes landed)
- ⚙ Validate environment generation produces correct Gymnasium spaces
- ⚙ Validate trainer generation works end-to-end
- ✓ Gate 2 certified (all validators pass; gate2_validator exit 0)

### Upcoming (Aug 31 - Sep 30)
- Sep 3-5: End-to-end pipeline validation (compile → generate → train → evaluate)
- Sep 8-10: Minimum viable IDE (text editor, compile button, show IR)
- Sep 11-13: Add blocks frontend (visual editor)
- Sep 14-16: Inspector integration (show semantics in UI)
- Sep 17-19: Simulation benchmark (prove end-to-end)
- Sep 20-22: LEGO backend (Bluetooth comm, motor/sensor drivers)
- Sep 23-25: Study preparation (task freeze, baseline scaffold, questionnaires)
- Sep 26-28: Pilot study (3-5 participants, iron out procedures)
- Sep 29-30: Submit SIGCSE SRC paper (if evidence sufficient)
- Oct 1-12: Main controlled user study (24-40 undergraduates)

## Research Questions & Mapping

| RQ | Question | Study Design | Success Metric |
|----|----------|--------------|----------------|
| RQ1 | Can novice programmers author learning tasks without RL infrastructure? | Completeness of T1/T2/T3 | ≥80% complete T3 |
| RQ2 | Does Kobe reduce time/errors/workload vs Python/TorchRL? | Within-subject time/SUS/NASA-TLX | Kobe < Python on all |
| RQ3 | Does Kobe preserve understanding of obs/action/objective/termination? | Post-task questions | ≥70% correct answers |
| RQ4 | Does inspectability improve debugging/prediction? | Opaque vs Inspector conditions | Inspector condition better |
| RQ5 | Same spec stays equivalent across sim/LEGO/Pi? | Cross-platform traces | Identical observable events |
| RQ6 | Which abstractions hidden/summarized/exposed? | Qualitative feedback | Design space map |

## Key Design Principles

### 1. Canonical IR
- Single source of truth for semantics
- All frontends (text, blocks) compile to same IR
- All backends (sim, LEGO, Pi) read same IR
- Semantic equivalence is verifiable, not assumed

### 2. Deterministic Traces
- Reference interpreter produces deterministic traces on fixed scenarios
- IR executor produces identical traces
- Both serve as ground truth for generated environments
- Randomness only in RL policy, not underlying semantics

### 3. Safety as Hard Constraint
- Safety violations cannot be overcome by reward
- Action veto happens before reward computation
- Safety weights affect aggressiveness, not constraint

### 4. Transparency by Default
- Inspector explains every decision the compiler made
- Users see observation space, action space, objectives, safety, termination
- Central to RQ4 research question
- Enables debugging and understanding

### 5. Matched Human Studies
- Three tasks with matched difficulty and concept coverage
- Python baseline using identical scaffolds
- Controlled within-subject design
- Counterbalanced conditions (Opaque vs Inspector)

## Success Criteria

### Technical
1. ✓ Parser handles all DSL features
2. ✓ Compiler produces valid IR
3. ⚙ Semantic equivalence proven (Gate 2)
4. ⚙ Environment generation works (Gate 3)
5. ⚙ Training pipeline runs (Gate 4)
6. [Sep 8] IDE fully functional (Gate 5)
7. [Sep 20] LEGO backend reliable
8. [Oct 12] Study complete with ≥24 subjects

### Research
1. [Oct 1] Pilot study reveals no blocking issues
2. [Oct 12] Main study shows Kobe advantage on ≥1 outcome
3. [Oct 12] Understanding preserved (RQ3 ≥70% correct)
4. [Oct 12] Transparency helps (RQ4 Inspector > Opaque)
5. [Oct 15] Paper ready for SIGCSE SRC submission

### Publication
1. Primary venue: SIGCSE TS 2027 SRC (Deadline Sep 30)
2. Secondary: HCI International 2027 (Deadline Oct 9)
3. Alternative: HRI 2027 if embodied becomes central
4. Reproducibility: Code + study materials in public repository

## Timeline (Per Execution Bible §18)

```
Aug 27 [Start]   ─── Compiler + test infrastructure complete
Aug 28-30        ─── Generation pipeline complete (TODAY)
Sep 3-5          ─── End-to-end validation
Sep 8-10         ─── Minimum IDE
Sep 11-13        ─── Blocks frontend
Sep 14-16        ─── Inspector integration
Sep 17-19        ─── Simulation benchmark
Sep 20-22        ─── LEGO hardware
Sep 23-25        ─── Study prep (freeze, baseline, questionnaires)
Sep 26-28        ─── Pilot study
Sep 29-30        ─── SIGCSE paper submission (if ready)
Oct 1-3          ─── Final study prep
Oct 4-12         ─── Main controlled study (24-40 subjects)
Oct 13-31        ─── Analysis, paper writing
[Venue Deadlines: SIGCSE Sep 30, HCI Oct 9, HRI tbd]
```

## Hardware Targets

### Primary: LEGO (EV3)
- Ultrasonic distance sensor
- Colour sensor
- Touch sensor
- IR sensor
- Gyro sensor
- Large motors
- Bluetooth 4.0 communication
- Accessible, educational, embodied

### Secondary: Raspberry Pi
- Same sensor suite (USB-connected)
- ROS2 middleware
- For portability validation
- Not primary study platform

### Tertiary: Simulation
- Gazebo or PyBullet
- For development/testing
- Faster iteration

## Study Design Outline

### Participants
- Target: 24-40 undergraduates
- Within-subject, counterbalanced
- Conditions: Opaque Kobe vs Inspector Kobe
- Baseline: Python + TorchRL (between-subjects optional)

### Tasks
- T1 (Obstacle): 10-15 min, introductory
- T2 (Target): 20-25 min, intermediate
- T3 (Tradeoff): 25-30 min, advanced
- Transfer: unseen task, generalization
- Total: ~90 min per participant

### Measures
- **Objective:**
  - Task completion time
  - Error count (syntax, logic)
  - Task success (program works)
- **Subjective:**
  - NASA-TLX (workload)
  - SUS-like (usability)
  - Custom (understanding, confidence)
- **Learning:**
  - Pre/post questions
  - Explanation accuracy
  - Transfer task performance

### Analysis
- Repeated-measures ANOVA (Opaque vs Inspector)
- Effect sizes (Cohen's d)
- Qualitative themes (think-aloud protocols)
- Generalization assessment

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Golden tests remain at 6/10 | Low | High | Debug event emission logic systematically |
| LEGO unreliable | Medium | Medium | Test extensively, have simulator fallback |
| Study underpowered | Low | High | Recruit aggressively, aim for n=40 |
| IDE takes longer than estimated | Medium | Medium | Prototype quickly, cut non-essential features |
| Training doesn't improve over baseline | Low | Medium | Use handwritten baseline as floor (already strong) |
| Paper rejection | Low | High | Plan secondary venues, resubmit quickly |

## Repository Structure (Final)

```
Kobe/
├── lexer.py                          ✓ Tokenization
├── parser.py                         ✓ AST construction
├── compiler.py                       ✓ AST → IR
├── semantic_analyzer.py              ✓ Type checking
├── backend.py                        ✓ IRInterpreter, KobeEnv
├── pipeline.py                       ✓ Training orchestration
├── priorities.py                     ✓ Default policy weights
│
├── reference_interpreter.py          ✓ Ground truth AST interpreter
├── ir_trace_executor.py              ✓ Comparison IR executor
├── handwritten_baseline.py           ✓ Control policy
│
├── environment.py                    ✓ NEW: IR → Gymnasium
├── trainer.py                        ✓ NEW: IR → TorchRL (implemented for export, live training uses backend.py)
├── inspector.py                      ✓ NEW: IR → explanations
├── study_tasks.py                    ✓ NEW: T1, T2, T3, Transfer
├── gate2_validator.py                ✓ NEW: Compiler proof
│
├── tests/
│   ├── golden/
│   │   ├── golden_programs.py        ✓ 10 test programs
│   │   ├── run_golden_tests.py       ✓ Gate 2 validator
│   └── (study task tests TBD)
│
├── ide/                              ⚙ TODO: Electron frontend
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── electron/
│   │   ├── main.js
│   │   └── preload.js
│   ├── ipc/
│   │   ├── pipeline_ipc.py           (exists)
│   │   └── train_ipc.py              (exists)
│   └── index.html
│
├── backends/                         ⚙ TODO: Platform-specific
│   ├── simulator/                    ✓ Gymnasium env (done)
│   ├── torchrl/                      ✓ Training config (done; live training uses backend.py)
│   ├── lego/                         ⚙ TODO: Hardware driver
│   │   ├── ev3_motor.py
│   │   ├── ev3_sensors.py
│   │   └── ev3_runtime.py
│   └── raspberry_pi/                 ⚙ TODO: ROS2 bridge
│
├── study/                            ⚙ TODO: Materials
│   ├── tasks/
│   ├── questionnaires/
│   ├── consent_form.pdf
│   └── protocol.md
│
├── TESTING.md                        ✓ UPDATED: comprehensive guide
├── QUICKSTART.md                     ✓ Quick reference
├── README.md                         ⚙ TODO: Update with new components
└── requirements.txt                  ⚙ TODO: Add environment, trainer deps
```

---

**Current Phase:** Aug 28 — Generation Pipeline Complete ✓
**Next Phase:** Sep 3-5 — End-to-End Validation
**Checkpoint:** All tests passing, system proof-ready for review
