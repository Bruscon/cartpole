# Repository Guidelines

## Project Structure
Flat Python project for CartPole reinforcement learning research. Core files:
- `cartpole.py` — main training loop (vectorized envs, custom rewards, eval scheduling)
- `SACAgent.py` — Soft Actor-Critic with discrete actions, twin Q-networks, prioritized replay
- `DQNAgent.py` — DQN agent (available but not currently wired as default)
- `evaluation_worker.py` — subprocess that loads policy model and runs eval episodes
- `TFPrioritizedReplayBuffer.py`, `sumtree.py` — GPU-accelerated replay buffer
- `TrainingLogger.py` — metrics logging (CSV, JSONL, PNG plots)
- `codex/` — experiment orchestration (program.md, results.tsv, TASK.md)

## Development
```bash
source /home/nick/rl-env/bin/activate
python cartpole.py --max-seconds 300 --log-dir logs/dev_run
```

## Git Workflow for Experiments
Every experiment gets its own branch:
1. Branch from master: `git checkout -b exp/<name>`
2. Implement changes, commit on the branch
3. Merge to master: `git merge --no-ff exp/<name>`
4. If experiment fails: `git revert <merge-commit>` on master
5. Never delete experiment branches — they're research history

## Coding Style
4-space indentation, `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
Keep changes focused. Test with a short training run before committing.

## Output Format
All stdout is structured JSON lines (see cartpole.py `emit()` function).
Do not print unstructured text to stdout — it breaks the JSON parsing pipeline.
