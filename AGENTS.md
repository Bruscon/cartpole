# Repository Guidelines

## Project Structure
Flat Python project for CartPole reinforcement learning research. Core files:
- `cartpole.py` — main training loop (vectorized envs, custom rewards, n-step accumulation, eval scheduling)
- `SACAgent.py` — Soft Actor-Critic with discrete actions, dueling Q-nets, LayerNorm, orthogonal init, prioritized replay
- `DQNAgent.py` — DQN agent (available but not currently wired as default)
- `evaluation_worker.py` — subprocess that loads policy model and runs eval episodes
- `TFPrioritizedReplayBuffer.py`, `sumtree.py` — GPU-accelerated replay buffer (supports variable batch sizes)
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

Keep git operations simple. Avoid cherry-pick chains — if a previous experiment's changes are needed, just re-implement them directly on a clean branch. Merge conflicts waste context.

## Coding Style
4-space indentation, `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants.
Keep changes focused. Test with a short training run or smoke test before committing.

## Keras / TF Gotchas
- **Keras 3**: Do NOT use `Lambda` layers with multiple inputs — they break on serialization. Use a custom `Layer` subclass instead (see `DuelingCombine` in SACAgent.py).
- **Orthogonal init**: Use float literals for gain (`gain=1.4142135`), not `tf.sqrt(2.0)`. TF tensor values in initializer configs break model save/load in the eval worker.
- **KerasTensors**: You cannot use raw `tf.*` operations on KerasTensors in Keras Functional API. Use `keras.ops.*` or wrap in a custom Layer subclass.

## Output Format
All stdout is structured JSON lines (see cartpole.py `emit()` function).
Do not print unstructured text to stdout — it breaks the JSON parsing pipeline.

## Diagnostics
Each training run writes rich logs to its `--log-dir`:
- JSONL with per-step metrics (loss, Q-values, alpha, LR)
- CSV summary
- PNG plots
- Eval results as JSON lines on stdout

When debugging, READ THESE LOGS. Don't just look at convergence/DNF — understand *why* a run failed by examining the training trajectory.
