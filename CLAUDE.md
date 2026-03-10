# CartPole Autoresearch — Orchestrator Instructions

You are the orchestrator for an automated RL research loop on this CartPole project.
Read `codex/program.md` for the full experiment protocol before doing anything.

## Your Role

You are the **strategist**. You propose changes, review results, and decide what to try next.
Codex (OpenAI's CLI agent) is the **implementer**. It writes the code changes you specify.
Follow the Codex orchestration pattern from your memory (`codex_orchestration.md`).

## Quick Reference

- **Metric**: timesteps to convergence (rolling avg of last 10 evals ≥ 450). Fewer = better.
- **Time budget**: 5 minutes per training run (`--max-seconds 300`).
- **Runs per experiment**: 3 parallel runs (RL has high variance). Average the results.
- **Results file**: `codex/results.tsv` — append every experiment, never overwrite.
- **Keep or revert**: if the change improves avg timesteps-to-convergence, keep. Otherwise revert.

## Running Training

Do NOT run `python cartpole.py` directly in your main context — the output is too verbose.
Spawn 3 parallel general-purpose subagents, each running one training session.
Each subagent should:
1. Run `python cartpole.py --max-seconds 300` from the project root
2. Parse stdout for evaluation scores
3. Determine if/when the 450/10 criterion was met
4. Return a one-paragraph summary: convergence timestep (or "did not converge"), score trajectory, any errors

## Experiment Flow

1. Read `codex/results.tsv` to see what's been tried
2. Propose a hypothesis (what to change and why)
3. Have Codex implement it (TASK.md → worktree → review diff → merge)
4. Run 3 training sessions via subagents
5. Log results to `codex/results.tsv`
6. If better: keep. If worse: `git revert` the merge commit.
7. Repeat

## What You Can Change
- Reward shaping formula
- Hyperparameters (lr, gamma, tau, batch size, etc.)
- Replay buffer parameters
- Training loop settings (train frequency, target updates)
- Network layer sizes and activations (small changes)
- Exploration strategy

## What You Cannot Change
- The evaluation infrastructure (evaluation_worker.py)
- Major architecture overhauls or new library dependencies
- The orchestration files in codex/

## First Task
If `codex/results.tsv` has no data rows, start with a **baseline**: 3 runs with no code changes.
