# Repository Guidelines

## Project Structure & Module Organization
This repository is a flat Python project centered on CartPole reinforcement learning experiments. Core training entry points live at the repo root: `cartpole.py` runs DQN training, `evaluation_worker.py` handles evaluation in a separate process, and `model_viewer.py` replays trained agents. Agent and buffer implementations are split across `DQNAgent.py`, `SACAgent.py`, `TFReplayBuffer.py`, `TFPrioritizedReplayBuffer.py`, `sumtree.py`, and `TrainingLogger.py`. Generated outputs go to `logs/` and `saved_models/`; media assets and notes live in `media/` and `notes/`.

## Build, Test, and Development Commands
Set up a virtual environment and install dependencies with `python3 -m venv rl-env && source rl-env/bin/activate && pip install -r requirements.txt`. Create a starter model with `python create_initial_model.py`. Run the main training loop with `python cartpole.py --log-dir logs/dev_run --model saved_models/initial_model.keras`; use `./run_cartpole_training.sh` for the default timestamped workflow. Replay a saved model with `python model_viewer.py --model saved_models/initial_model.keras`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for module-level constants such as `MAX_ANGLE`. Keep modules focused on one responsibility and prefer explicit CLI flags via `argparse` for runnable scripts. No formatter or linter is configured here, so keep imports grouped, comments short, and changes consistent with surrounding code.

## Testing Guidelines
There is no automated test suite yet. For code changes, run at least a smoke test: `python create_initial_model.py`, then a short `python cartpole.py --log-dir logs/test_run --model saved_models/initial_model.keras`, and confirm expected artifacts are written under `logs/`. For viewer or evaluation changes, validate with `python model_viewer.py --model <path>`. If you add tests, prefer `pytest` with files named `test_*.py`.

## Commit & Pull Request Guidelines
Recent history uses short, direct subjects such as `updated readme`, `adding gif`, and `new picture`. Keep commit messages concise, present tense, and specific to one change. Pull requests should summarize behavior changes, list the commands used for validation, and include updated plots or screenshots when training behavior, logging, or visual output changes.

## Artifacts & Configuration
Do not commit large generated contents from `logs/`, transient model checkpoints, or machine-specific environment paths unless they are intentional release artifacts. Document GPU, CUDA, or TensorFlow assumptions in the PR when they affect reproducibility.
