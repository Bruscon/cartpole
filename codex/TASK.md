## Experiment: exp17_layernorm_lr02

### Hypothesis
LayerNorm stabilizes training (smoother loss landscape), enabling a moderate LR increase from 0.01→0.02. exp16 showed LayerNorm+LR 0.03 was fast (675 avg) but unreliable (2/5). LR 0.02 should retain most of the speed benefit while improving reliability to 3+/5.

### Changes
- `SACAgent.py` — `_build_q_model()` and `_build_policy_model()`: Add LayerNormalization after each Dense(64) layer (Dense→LayerNorm→ReLU pattern)
- `SACAgent.py` — `__init__()`: learning_rate 0.01→0.02

### Expected mechanism
LayerNorm normalizes activations, reducing internal covariate shift and smoothing the loss landscape. This allows larger learning rates without divergence. LR 0.02 is a 2x increase (vs 3x in exp16) — should be aggressive enough to speed convergence but stable enough to avoid the DNFs seen at 0.03.

### Rollback plan
git revert the merge commit
