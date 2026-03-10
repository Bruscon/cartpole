# CartPole Autoresearch Loop

An automated RL research loop. Claude is the strategist. Codex is the implementer.
The goal: converge to perfect CartPole scores as fast as possible (fewest training steps).

## The Loop

1. **Claude proposes** an experiment — a structured spec describing what to change, why, and what to measure
2. **Codex implements** it in a git branch (branch per experiment, always)
3. **Claude reviews** the diff before merging to master
4. **3-5 parallel training runs** measure the result (RL has high variance)
5. **Claude decides**: keep (merge stays) or revert (`git revert` the merge commit)
6. **Claude logs** results and lessons learned to `codex/results.tsv`
7. Repeat with next hypothesis, informed by all previous results

## Convergence Criterion

- Eval runs **5 episodes per checkpoint**, reports the **average score**
- Convergence = rolling average of last **10 eval checkpoints ≥ 475** (each checkpoint is already a 5-episode mean)
- Metric: **training steps to convergence** (fewer = better)
- A run that never converges within 180s wall-clock = DNF
- **3-5 runs per experiment** (start with 3, run more for ambiguous results). Report: how many converged, average convergence step for those that did

## Experiment Spec Format

When proposing an experiment, Claude writes a spec like this:

```
## Experiment: <name>
### Hypothesis
<What you expect to happen and why — ground it in RL theory>

### Changes
<Bullet list of concrete code changes. Be specific enough that Codex can implement without ambiguity.>

### Expected mechanism
<Why should this help? What's the causal chain?>

### Rollback plan
<What to revert if it fails — usually just "git revert the merge commit">
```

Claude writes this spec to `codex/TASK.md`. Codex reads it and implements.

## What Claude Can Propose

**Anything that might help convergence speed.** This includes:

### Architecture
- Network topology (dueling heads, noisy layers, skip connections, attention)
- Activation functions (ReLU, GELU, Swish, etc.)
- Layer normalization, batch normalization
- Weight initialization strategies (orthogonal, He, Xavier)
- Separate network sizes for different components

### Algorithm
- N-step returns
- Distributional RL (C51, QR-DQN)
- Different SAC variants
- Double Q-learning variants
- Switching between SAC and DQN (both exist in codebase)
- Mixing algorithms (e.g., SAC with distributional critics)

### Training
- Learning rate schedules (cosine, warmup+decay, cyclical)
- Replay buffer variants (uniform, PER tuning, hindsight)
- Training frequency, gradient steps per env step
- Batch size, buffer size, train_start threshold
- Target network update strategies

### Reward Shaping
- Any function of the state variables
- Reward clipping, normalization, scaling
- Potential-based shaping

### Exploration
- Entropy bonuses (SAC temperature tuning)
- Noisy networks
- Epsilon schedules
- Boltzmann exploration

### Dependencies
- **pip install is allowed**. If a library helps (e.g., `tensorflow-probability`, `gymnasium[classic_control]`, etc.), Codex can install it
- Add new deps to requirements.txt

## What Claude Cannot Change
- The orchestration files in `codex/` (program.md, results tracking)
- The fundamental eval metric (score = steps survived, max 500)

## Multi-Change Experiments

Claude is encouraged to propose **compound changes** when there's theoretical reason to believe they interact. Examples:
- "Layer norm enables higher learning rates" → change both together
- "Dueling architecture + n-step returns" → known to be complementary (Rainbow)
- "Orthogonal init stabilizes training" → enables previously-unstable experiments to work
- "Smaller network + noisy layers" → trade capacity for better exploration

Single-variable experiments are fine for isolating effects, but don't be afraid to propose bigger moves.

**Revisit failed experiments after stabilization.** A change that was unstable before (e.g., n-step returns) might work perfectly after you fix the underlying instability (e.g., with orthogonal init).

## Diagnose Before Guessing

**When multiple experiments fail with the same pattern, STOP and diagnose the root cause.**

Don't burn experiments on blind hyperparameter sweeps. Instead:
1. Read training logs from failed runs (JSONL in log-dir, eval lines on stdout)
2. Look at the *trajectory*: when does the agent fail? Is it never learning, or learning then collapsing?
3. Check Q-values, loss, alpha, episode lengths — is something diverging or collapsing?
4. Form a mechanistic hypothesis about *why* it fails, then design a targeted experiment

This approach is what found the orthogonal init breakthrough in CartPole — 8 experiments of hyperparameter sweeping failed, but diagnosing the failure mode (init variance → catastrophic collapse) solved it in one shot.

## Git Discipline

**Every experiment gets a branch.** This is non-negotiable.

1. Codex creates branch `exp/<name>` from master
2. Codex commits changes on that branch
3. Codex merges to master with a merge commit (not fast-forward: `--no-ff`)
4. If the experiment fails: `git revert <merge-commit>` on master
5. The experiment branch stays forever — it's our research history

Keep git operations simple. Avoid cherry-pick chains across branches — they cause merge conflicts and waste context. If you need changes from a previous experiment, just re-implement them directly on a clean branch.

## Results Tracking

File: `codex/results.tsv`
Columns: `experiment | description | converged | avg_step | run_details | status | lesson`

- `converged`: "3/5", "5/5", etc.
- `avg_step`: average convergence step for runs that converged (DNF if 0/5)
- `run_details`: per-run convergence steps, comma-separated
- `status`: `keep` or `revert`
- `lesson`: 1-2 sentence takeaway that informs future experiments

## Running Training

**Resource-aware parallel runs.** System resources (GPU memory, RAM) may be constrained.

Claude spawns **1 background subagent** that launches training runs in parallel.
The subagent should:
1. Check system resources first (`nvidia-smi`, `free -h`) to determine safe parallelism (start with 3)
2. Launch N training processes in parallel (background bash), each with a unique `--log-dir /tmp/expN_runM`:
   `source /home/nick/rl-env/bin/activate && cd /home/nick/Documents/cartpole && python cartpole.py --max-seconds 180 --log-dir /tmp/<exp>_run<N>`
3. Wait for all N to finish (~3 minutes).
4. For each run, parse the last `{"type": "done", ...}` line. If it has `"convergence_step": N`, converged at step N. Otherwise DNF.
5. Return: per-run convergence steps (or DNF), resource usage observed, and any errors.

Convergence detection is built into `cartpole.py` — the subagent does NOT need to recompute it.
While the subagent runs in background, Claude should plan the next experiment (write spec, launch Codex).

## Baseline

Check `codex/results.tsv` for the latest baseline row.
Skip re-baselining unless the eval infrastructure code has changed.
