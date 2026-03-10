# CartPole Autoresearch Loop

An automated RL research loop. Claude is the strategist. Codex is the implementer.
The goal: converge to perfect CartPole scores as fast as possible (fewest training steps).

## The Loop

1. **Claude proposes** an experiment — a structured spec describing what to change, why, and what to measure
2. **Codex implements** it in a git branch (branch per experiment, always)
3. **Claude reviews** the diff before merging to master
4. **5 parallel training runs** measure the result (RL has high variance)
5. **Claude decides**: keep (merge stays) or revert (`git revert` the merge commit)
6. **Claude logs** results and lessons learned to `codex/results.tsv`
7. Repeat with next hypothesis, informed by all previous results

## Convergence Criterion

- Eval runs **5 episodes per checkpoint**, reports the **average score**
- Convergence = rolling average of last **10 eval checkpoints ≥ 475** (each checkpoint is already a 5-episode mean)
- Metric: **training steps to convergence** (fewer = better)
- A run that never converges within 180s wall-clock = DNF
- **5 runs per experiment**. Report: how many converged, average convergence step for those that did

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
- Batch size, buffer size
- Target network update strategies

### Reward Shaping
- Any function of the 4 state variables (cart_pos, cart_vel, pole_angle, pole_angular_vel)
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
- The fundamental eval metric (score = steps survived in CartPole-v1, max 500)

## Multi-Change Experiments

Claude is encouraged to propose **compound changes** when there's theoretical reason to believe they interact. Examples:
- "Layer norm enables higher learning rates" → change both together
- "Dueling architecture + n-step returns" → known to be complementary (Rainbow)
- "Smaller network + noisy layers" → trade capacity for better exploration

Single-variable experiments are fine for isolating effects, but don't be afraid to propose bigger moves.

## Git Discipline

**Every experiment gets a branch.** This is non-negotiable.

1. Codex creates branch `exp/<name>` from master
2. Codex commits changes on that branch
3. Codex merges to master with a merge commit (not fast-forward: `--no-ff`)
4. If the experiment fails: `git revert <merge-commit>` on master
5. The experiment branch stays forever — it's our research history

This means we can always `git log --oneline --graph` and see every experiment tried.

## Results Tracking

File: `codex/results.tsv`
Columns: `experiment | description | converged | avg_step | run_details | status | lesson`

- `converged`: "3/5", "5/5", etc.
- `avg_step`: average convergence step for runs that converged (DNF if 0/5)
- `run_details`: per-run convergence steps, comma-separated
- `status`: `keep` or `revert`
- `lesson`: 1-2 sentence takeaway that informs future experiments

## Running Training

Claude spawns **1 background subagent** that launches all 5 training runs in parallel.
The subagent should:
1. Launch 5 training processes in parallel (background bash), each with a unique `--log-dir /tmp/expN_runM`:
   `source /home/nick/rl-env/bin/activate && cd /home/nick/Documents/cartpole && python cartpole.py --max-seconds 180 --log-dir /tmp/<exp>_run<N>`
2. Wait for all 5 to finish (~3 minutes).
3. For each run, parse the last `{"type": "done", ...}` line. If it has `"convergence_step": N`, converged at step N. Otherwise DNF.
4. Return: per-run convergence steps (or DNF), and any errors.

Convergence detection is built into `cartpole.py` — the subagent does NOT need to recompute it.
While the subagent runs in background, Claude should plan the next experiment (write spec, launch Codex).

## Baseline

Current Phase 2 baseline: **0/5 DNF** (peak rolling avgs 242-414)
Config: policy net 64-64, tau=0.1, batch_size=256, gamma=0.999, lr=0.01

The agent learns but can't sustain 475+ 5-episode mean in 180s. Policy instability
(scores oscillate between 150-500 late in training) is the main bottleneck.
Architectural changes (layer norm, dueling, etc.) are needed to stabilize and speed convergence.

Skip re-baselining unless the eval infrastructure code has changed. Check `codex/results.tsv`
for the most recent baseline row.
