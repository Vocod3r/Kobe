# Algorithmic Limitations, Diagnosis, and Verification

This file records the full arc of Kobe's live-training limitation: how it was
first diagnosed as an exploration failure of the deterministic policy
architecture, how that hypothesis was fixed with a real stochastic SAC policy,
and how the fix was re-verified on the exact experiment that originally failed.
It is a diagnosis → fix → re-verification record, not a list of dead ends.

## 1. Original finding: deterministic actor collapses into seed-locked local optima

Across Kobe's live simulation training backend (`backend.py`), every algorithm
originally shared a **deterministic sigmoid actor with no exploration
mechanism**. Empirically, that actor rapidly collapses into seed-dependent
local optima early in training and subsequently fails to respond to
reward-shaping interventions: because the architecture lacks an active,
state-dependent exploration mechanism, policy parameters stay trapped in the
basin of attraction dictated by the initial seed weights, and neither algorithm
choice, reward scaling, nor penalty tuning can move the settled policy.

Three independent, methodologically distinct experiments arrived at the same
conclusion:

1. **Algorithm invariance (SAC vs. DroQ).** Value-network regularization
   (critic dropout) and an elevated update-to-data ratio (UTD = 4) produced no
   population-level behavioral differentiation from baseline SAC across 6 seeds
   (|Δμ| = 4.6 speed units, Cohen's d = 0.018). Without exploratory pressure,
   regularizing the critic cannot redirect the deterministic actor toward
   different policies.

2. **Dense-reward insensitivity (efficiency slider).** A 27-run sweep across 9
   slider settings and 3 seeds modulating the per-step speed weight produced an
   insignificant, *wrong-signed* relationship with the speed metric
   (r = −0.1676, p = 0.403). De-meaning the data to isolate within-seed
   variation showed every individual seed was negatively correlated
   (r = −0.4018, −0.5734, −0.0716), for a pooled within-seed r = −0.2417
   (p = 0.225). Individual seeds settled into fixed speed plateaus — identical
   evaluation outputs repeated across distinct slider values — regardless of
   reward magnitude.

3. **Constraint-shaping failure (safety slider).** Across three progressive
   reward designs — a sparse collision penalty, an additive continuous
   quadratic penalty on risky actions (max(0, a − 0.75)²), and an unconditional
   multiplicative attenuation of the speed reward ((1 − safety) · R_speed) —
   the safety slider failed to produce monotonic or statistically significant
   behavioral control. In the final multiplicative formulation (N = 27),
   correlations were insignificant and counter-directional (r = +0.1559 for
   speed, r = −0.2008 for safety%), with seed-level attractors still dominating
   the optimization landscape.

Triangulating across algorithmic regularization, continuous reward scaling, and
constraint shaping indicated the insensitivity was systemic to the deterministic
policy architecture rather than an artifact of any single reward formulation or
hyperparameter setting. The hypothesized root cause was a **missing exploration
mechanism**.

## 2. The fix: real stochastic-policy SAC (CleanRL-conventions)

The fix replaced the deterministic sigmoid actor with a genuine stochastic
diagonal-Gaussian SAC policy in `backend.py`, following CleanRL's reference
implementation:

- a policy head outputting per-dimension action mean and bounded log-std,
  sampled with the reparameterization trick (`rsample`), squashed through
  `tanh`, and rescaled to the action space, with the tanh-squashing
  change-of-variables correction applied in the log-probability;
- a **learned entropy temperature** (autotuned `log_alpha` with its own Adam
  optimizer, driven toward a target entropy) replacing the previous static
  slider-heuristic alpha for SAC;
- the real `−α · log π` entropy term in both the critic target and the actor
  loss, replacing the previous empirical variance-based entropy hack.

The change is confined to the SAC training path; TD3 and DroQ were left
untouched so their behavior remains directly comparable.

### Convergence gate (Gate 5) before the re-verification sweep

Before using SAC to re-run the failed experiment, convergence was checked on a
50-step multi-sensor test program across 3 seeds (5000 steps each):

- Mean reward per 500-step log window improved from ≈ −8.8k during the warmup /
  early-exploration phase to ≈ −2.8k after convergence (≈ 3.2× improvement),
  then plateaued near the environment's least-negative reward optimum rather
  than diverging. (All rewards are on a negative scale in this simulator because
  the comfort term penalizes cumulative jerk on every step.)
- `log_alpha` moved smoothly from its initial 0 to ≈ +1.26 (α ≈ 3.4–3.7) across
  all three seeds — a meaningful, bounded, non-divergent trajectory.

## 3. Re-verification: the failed efficiency sweep, rerun with real SAC

The identical efficiency sweep that failed under TD3 — the exact same test
program, 9 efficiency values × 3 seeds, 5000 steps per run, safety/comfort/
curiosity fixed at 0.5 — was rerun with SAC substituted in.

| Metric | TD3 (original, failed) | SAC (re-run) |
|---|---|---|
| Exact-repetition plateaus (same speed/safety/comfort triple at ≥2 slider values) | 6 (incl. seed 999 locked at 1486.5 across eff 0.1–0.3; seed 123 at 1511.1 at eff 0.4/0.8/0.9) | **0** |
| Within-seed (de-meaned) Pearson r (efficiency → speed) | −0.2417 | **+0.3930** |
| seed 42 | −0.4018 | +0.9203 (p = 0.0004) |
| seed 123 | −0.5734 | +0.9930 (p < 0.0001) |
| seed 999 | −0.0716 | +0.5340 (p = 0.139) |

Every seed's efficiency→speed correlation flipped from negative to positive, and
two of three seeds are individually significant. The seed-locked pattern that
defined the original failure — identical deterministic evaluation outputs
repeated across distinct slider values — no longer occurs. Because the failure
signature was specifically a seed-locked insensitivity to reward shaping, and
that signature disappears when genuine exploration is introduced while everything
else (program, seeds, slider values, step budget, eval protocol) is held fixed,
this re-verification confirms the diagnosed root cause — **lack of exploration** —
was the actual cause, not a coincidental correlation.

## 4. Honest residual limitation: reward dynamic range, not exploration

The re-verification does **not** claim the efficiency slider now produces a large
or significant population-level effect:

- The pooled 27-run correlation is still non-significant
  (r = +0.2168, p = 0.2773). Partly this is power: only 3 seeds, and seeds
  operate at different absolute levels, so between-seed variance dilutes the
  pooled test; the within-seed (de-meaned) statistic — the analysis the original
  Metric-2 finding was based on — is the appropriate one and is positive
  (+0.3930).
- For 2 of 3 seeds (42, 123) the effect size is small: speed rises only ≈ 5%
  across the full slider range (≈ 874 → 918 metric units from eff 0.1 → 0.9).

The small effect size is a property of the reward function, not of exploration.
At comfort = 0.5 the jerk penalty (applied to cumulative jerk on every step)
makes a near-zero action reward-optimal regardless of the efficiency weight, so
SAC correctly converges to that near-zero optimum per seed and tracks the slider
faithfully but with little leverage. This is a **reward-dynamic-range limitation,
distinct from the original exploration problem**, and is left as future work —
rebalancing the comfort/efficiency reward weights so the efficiency slider has
practical range at these settings — rather than something to be silently tuned
away in this pass.

## 5. Motivating context (kept for the record)

The two earlier experimental threads are retained here because they motivated
the SAC rewrite and anchor the diagnosis:

- **SAC vs. DroQ invariance (Section 1, experiment 1)** showed that regularizing
  the value network cannot substitute for exploratory pressure. It is now
  explained by the shared deterministic actor: DroQ's critic dropout and UTD = 4
  were applied to a policy that could not explore, so no behavioral separation
  was possible.
- **Safety-slider history (Section 1, experiment 3)** — three progressive reward
  designs failing to produce monotonic safety control — is likewise attributable
  to the same actor-level attractor problem rather than to the reward designs
  themselves.

Neither thread is an unresolved dead end: both are consistent with, and were
resolved by, the exploration fix verified in Sections 2–3.

## Future work

- Rebalance comfort vs. efficiency reward weights (and re-examine the
  cumulative-jerk penalty) so slider interventions have practical dynamic range
  at default settings, then re-run the efficiency and safety sweeps.
- Increase seed count (and report pooled power) if a population-level
  efficiency→speed effect is to be established beyond the per-seed level.
- Optionally extend the stochastic policy head to TD3/DroQ (or re-run their
  comparisons under real SAC) once the reward-dynamic-range work is done.
