# CartPole Autoresearch Loop

Inspired by Karpathy's autoresearch pattern but with Claude as the strategist.

## The Loop

1. **Claude proposes** a change (reward shaping, hyperparams, training tricks — NOT major architecture or library changes)
2. **Codex implements** it via the standard orchestration pattern (TASK.md → worktree → review → merge)
3. **3 parallel Claude subagents** each run `python cartpole.py --max-seconds 300`, extract the convergence metric, return a one-paragraph summary
4. **Claude decides** keep or revert based on the 3 summaries, logs to `codex/results.tsv`
5. Repeat with next hypothesis

## Convergence Criterion ("better")
- Rolling average of last **10 evaluation runs ≥ 450**
- Metric tracked: **timesteps to convergence** (fewer = better)
- A run that never hits 450/10 within 300s = failure

## Results Tracking
File: `codex/results.tsv`
Columns: `experiment | description | run1_timesteps | run2_timesteps | run3_timesteps | avg | status | notes`
Status: `keep` or `revert`

## Scope of Changes Claude Can Propose
- Reward shaping (the custom angle × position formula)
- Hyperparameters (lr, gamma, tau, batch size, epsilon schedule, etc.)
- Replay buffer parameters (alpha, beta, buffer size)
- Training loop tweaks (train frequency, target update frequency)
- Network size/activation (small changes, not wholesale rewrites)
- Exploration strategy

## Out of Scope
- Switching RL algorithm (SAC ↔ DQN is ok, but not new libraries)
- Major architecture overhauls
- Changing the evaluation infrastructure

## Subagent Pattern for Running Training
Spawn 3 parallel general-purpose subagents, each with a prompt like:
"Run `python cartpole.py --max-seconds 300` in the project root.
Monitor stdout. Report: did rolling avg hit 450 over 10 evals? If yes, at what timestep?
What was the score trajectory (every ~30s)? Any errors? Return a short summary only."

This protects orchestrator context from verbose training output.

## Baseline
First experiment is always a baseline (no code changes, just 3 runs) to establish
the current timesteps-to-convergence before any modifications.
