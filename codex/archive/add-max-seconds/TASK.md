## Objective
Add a `--max-seconds` command-line argument to `cartpole.py` that stops training cleanly after a given number of wall-clock seconds.

## Context
Read README.md for project context.

The training loop in `cartpole.py` currently only stops when `total_steps >= TOTAL_TIMESTEPS`. There is no time-based stopping. We need to cap training runs at 300 seconds (5 minutes) for automated experiments.

`timeout 300 python cartpole.py` is NOT acceptable — it sends SIGTERM which bypasses the normal shutdown path (`finally` block at the end of main), risking corrupt model checkpoints and orphaned evaluation subprocesses.

## Constraints
- Only modify `cartpole.py`
- Use `time.monotonic()` for the deadline check
- Check the deadline at the top of the main training while-loop so it exits cleanly
- The `--max-seconds` argument should be optional (default: None = no limit, existing behavior preserved)
- The clean shutdown must go through the existing `finally` block so models are saved and the eval worker is stopped properly
- Do not change any other behavior, hyperparameters, or logic

## Definition of Done
- `python cartpole.py --max-seconds 10` starts, runs for ~10 seconds, then exits cleanly through the normal shutdown path (models saved, eval worker stopped)
- `python cartpole.py` (no argument) behaves exactly as before
