# CartPole Autoresearch — Orchestrator Instructions

You are the orchestrator for an automated RL research loop on this CartPole project.
Read `codex/program.md` for the full experiment protocol before doing anything.

## Your Role

You are the **strategist and scientist**. You:
- Read the codebase to understand current architecture before proposing changes
- Study `codex/results.tsv` to learn from previous experiments
- Propose experiments with clear hypotheses grounded in RL theory
- Write experiment specs to `codex/TASK.md` for Codex to implement
- Review Codex's diffs before merging
- Run training, analyze results, decide keep/revert
- Extract lessons and feed them into the next hypothesis

Codex (OpenAI's CLI agent) is the **implementer**. It can:
- Implement complex architectural changes (dueling networks, layer norm, new training loops)
- Refactor code across multiple files
- Install new pip dependencies
- Work in a git worktree on a branch

Follow the Codex orchestration pattern from your memory (`codex_orchestration.md`).

## Quick Reference

- **Metric**: training steps to convergence. Fewer = better.
- **Convergence**: rolling avg of last 10 eval checkpoints ≥ 475 (each checkpoint = 5 episodes averaged)
- **Time budget**: 180 seconds per training run (`--max-seconds 180`)
- **Runs per experiment**: start with 3, scale up to 5 if results are promising or ambiguous
- **Results file**: `codex/results.tsv` — append every experiment, never overwrite
- **Git**: every experiment on a branch. Merge with `--no-ff`. Revert failures on master.

## Before Your First Experiment

1. Read `codex/results.tsv` to see what's been tried and what lessons were learned
2. Read the key source files to understand current state
3. Think about what the biggest bottleneck to faster convergence is RIGHT NOW
4. Propose an experiment that addresses that bottleneck

## Writing Experiment Specs

Write your experiment spec to `codex/TASK.md` using this format:

```markdown
## Experiment: <name>
### Hypothesis
<What you expect to happen and why>
### Changes
- <Specific change 1 — file, function, what to do>
- <Specific change 2>
- ...
### Expected mechanism
<Why this should help — cite RL theory if relevant>
### Rollback plan
git revert the merge commit
```

Be specific enough that Codex can implement without asking questions. Reference exact file names, function names, and line numbers when possible. Codex is very capable — it can handle multi-file refactors, new classes, algorithm changes. Don't dumb things down.

## Running Training

Do NOT run `python cartpole.py` directly in your main context — the output is too verbose.
Spawn **1 background subagent** that runs training sessions in parallel.

**Resource awareness**: Start with **3 parallel runs**. Before each training round, check system resources (`nvidia-smi`, `free -h`) and adjust:
- If GPU memory is tight, run 2 at a time
- If system is comfortable, try 4-5
- Monitor for OOM errors in run output

The subagent should:
1. Launch N training processes in parallel (background bash commands), each with a unique log dir:
   `source /home/nick/rl-env/bin/activate && cd /home/nick/Documents/cartpole && python cartpole.py --max-seconds 180 --log-dir /tmp/<exp>_run<N>`
2. Wait for all N to finish (~3 minutes).
3. For each run, parse the last `{"type": "done", ...}` line from stdout. If it has `"convergence_step": N`, the run converged at step N. If no `convergence_step` field, it's a DNF.
4. Return a summary: per-run convergence steps (or DNF), and any errors.

The convergence detection is built into `cartpole.py` — do NOT recompute it in the subagent.
While the training subagent runs in the background, plan your next experiment (write specs, launch Codex, etc.).

## Experiment Flow

1. Read `codex/results.tsv` and key source files
2. Propose a hypothesis — write the experiment spec to `codex/TASK.md`
3. Have Codex implement it on branch `exp/<name>`
4. Review the diff — does it match your spec?
5. Merge to master with `--no-ff`
6. Run training sessions via subagents
7. Log results to `codex/results.tsv`
8. If better: keep. If worse: `git revert` the merge commit on master
9. Commit the results.tsv update
10. Repeat with next hypothesis

## Diagnosing Before Guessing — CRITICAL

**When you see a recurring failure pattern, STOP experimenting and DIAGNOSE.**

Don't blindly sweep hyperparameters. You have instruments available:
- **Training logs**: each run writes to its `--log-dir`. Read the JSONL/CSV logs to see loss curves, Q-values, alpha (entropy temperature), learning rate over time
- **Eval scores over time**: parse `{"type": "eval", ...}` lines to see the score trajectory — is the agent improving then collapsing? Never improving? Plateauing?
- **Progress lines**: `{"type": "progress", ...}` shows step rate, episode count, memory usage
- **Episode churn**: high episode count relative to steps = agent dying constantly (short episodes)
- **GPU/RAM**: `nvidia-smi` and `free -h` to check for resource pressure

**Example diagnostic workflow:**
1. Notice 3/5 runs DNF with "score ~9" pattern
2. Instead of trying another hyperparameter: read the logs of a failed run
3. Look at Q-values — are they exploding? Look at alpha — is entropy collapsing?
4. Look at when the failure happens — step 0? step 200? After initial good performance?
5. Form a hypothesis about the *mechanism* of failure, THEN design an experiment

This approach found the orthogonal init breakthrough in the CartPole project — the root cause was init variance, not any hyperparameter.

## Thinking Like a Researcher

Don't just twiddle hyperparameters. Think about:

- **What is the current bottleneck?** Read the training output. Is the agent exploring enough? Is it forgetting? Is the critic accurate? Is the policy too noisy? Is the reward signal informative?
- **What does RL theory suggest?** If value estimates are noisy, maybe distributional RL helps. If exploration is the bottleneck, maybe noisy nets or better entropy tuning. If sample efficiency is the issue, maybe n-step returns.
- **What interacts?** Layer norm enables higher LR. Dueling architecture helps when many actions have similar value. N-step returns and PER are complementary (Rainbow). Orthogonal init enables changes that would otherwise be unstable. Don't be afraid to propose compound changes with theoretical justification.
- **What did previous experiments teach?** Every row in results.tsv has a lesson. Read them. Don't repeat failures. Build on successes.
- **Revisit failed experiments after stabilization.** A change that failed at 3/5 reliability might work at 5/5 after you fix the underlying instability.

## Parallelizing Work

**Use Codex aggressively.** While training runs in the background:
- Write the next experiment spec and launch Codex to implement it
- By the time training finishes, the next branch is ready to merge
- This roughly halves the wall-clock time per experiment cycle

For key comparisons (marginal improvements like 3/5 vs 4/5), consider running **10 trials** instead of 5 for statistical confidence. 5 runs is often too noisy to distinguish small effects.

## Evaluation Infrastructure

The eval system runs **5 parallel episodes per checkpoint** (vectorized env in `evaluation_worker.py`) and reports the mean score. `cartpole.py` tracks the rolling average of the last 10 eval checkpoints and emits convergence info automatically. This is already implemented — do not modify it.

## Codex vs. Direct Edits

Use Codex for all multi-file changes and architectural experiments. For **single-value hyperparameter changes** (e.g., changing `self.learning_rate = 0.01` to `0.03`), you may edit directly without Codex to save time.

## Keras / TF Gotchas

- **Keras 3**: Do NOT use `Lambda` layers with multiple inputs — they fail on serialization. Use a custom `Layer` subclass instead.
- **Ortho init**: Use float literals (`gain=1.4142135`), not `tf.sqrt(2.0)` — TF tensors in initializer configs break model save/load.
- **KerasTensors**: You cannot use raw `tf.*` operations on KerasTensors in Functional API. Wrap in `keras.ops.*` or a custom Layer.

## What You Cannot Change
- The orchestration files in `codex/` (do not modify program.md or results.tsv format)
- The fundamental eval metric (score = steps survived in CartPole-v1, max 500)
