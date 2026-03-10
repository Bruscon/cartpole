## Experiment: exp26_nstep_returns

### Hypothesis
3-step returns will improve convergence reliability (3/5→4+/5) by providing faster credit assignment. CartPole has delayed consequences (pole tilting takes several steps to become critical), and n-step returns propagate reward information 3x faster than 1-step TD.

### Changes

**`SACAgent.py` — `__init__()`**: Add n-step hyperparameter:
```python
self.n_step = 3
```

**`SACAgent.py` — `_compute_q_targets()`**: Change `self.gamma` to `self.gamma ** self.n_step`:
```python
# Change this line:
targets = rewards + (1.0 - dones_float) * self.gamma * next_values
# To:
targets = rewards + (1.0 - dones_float) * (self.gamma ** self.n_step) * next_values
```

**`cartpole.py` — main()`**: Add n-step accumulation logic. This is the main change.

After the line `agent = SACAgent(...)`, add:
```python
from collections import deque
n_step_buffers = [deque(maxlen=agent.n_step) for _ in range(n_envs)]
```

Replace the current experience storage block (line ~177, `agent.memory.add(states, actions, custom_rewards, next_states, dones)`):

```python
# N-step return accumulation
states_np = states.numpy() if hasattr(states, 'numpy') else states
actions_np = actions.numpy() if hasattr(actions, 'numpy') else actions
rewards_np = custom_rewards.numpy() if hasattr(custom_rewards, 'numpy') else custom_rewards
next_states_np = next_states.numpy() if hasattr(next_states, 'numpy') else next_states
dones_np = dones.numpy() if hasattr(dones, 'numpy') else dones

for i in range(n_envs):
    n_step_buffers[i].append((
        states_np[i], actions_np[i], rewards_np[i],
        next_states_np[i], dones_np[i]
    ))

    # Flush when buffer is full OR episode ended
    if len(n_step_buffers[i]) == agent.n_step or dones_np[i]:
        # Compute n-step discounted return
        G = 0.0
        for j in reversed(range(len(n_step_buffers[i]))):
            _, _, r_j, _, d_j = n_step_buffers[i][j]
            if d_j and j < len(n_step_buffers[i]) - 1:
                G = r_j  # Reset at terminal (earlier terminal in sequence)
            else:
                G = r_j + agent.gamma * G

        s_0, a_0, _, _, _ = n_step_buffers[i][0]
        _, _, _, s_n, d_n = n_step_buffers[i][-1]

        # Add single transition to buffer
        agent.memory.add_single(
            s_0, a_0, G, s_n, d_n
        )

        # If episode ended, clear the buffer; otherwise just pop the oldest
        if dones_np[i]:
            n_step_buffers[i].clear()
```

**`TFPrioritizedReplayBuffer.py` — add `add_single()` method**: The current `add()` method takes batched tensors from all envs. We need a method that adds a single transition:

```python
def add_single(self, state, action, reward, next_state, done):
    """Add a single transition to the buffer."""
    # Convert to tensors if needed
    state = tf.constant(state, dtype=tf.float32)
    action = tf.constant(action, dtype=tf.int32) if not isinstance(action, tf.Tensor) else action
    reward = tf.constant(reward, dtype=tf.float32)
    next_state = tf.constant(next_state, dtype=tf.float32)
    done = tf.constant(done, dtype=tf.bool)

    # Reshape to batch of 1
    state = tf.expand_dims(state, 0)
    action = tf.expand_dims(action, 0)
    reward = tf.expand_dims(reward, 0)
    next_state = tf.expand_dims(next_state, 0)
    done = tf.expand_dims(done, 0)

    self.add(state, action, reward, next_state, done)
```

Wait — the existing `add()` method may handle batched inputs. Check the current `add()` method signature and implementation. If it already handles variable batch sizes, you can just batch all the n-step transitions collected in one timestep and call `add()` once. That would be more efficient.

**Efficient batched approach** (preferred): Instead of calling `add_single()` per env, collect all n-step transitions for this timestep into lists, then batch them:

```python
# After the per-env loop, batch all collected transitions
if batch_states:  # If any transitions were collected
    agent.memory.add(
        tf.constant(batch_states, dtype=tf.float32),
        tf.constant(batch_actions, dtype=tf.int32),
        tf.constant(batch_rewards, dtype=tf.float32),
        tf.constant(batch_next_states, dtype=tf.float32),
        tf.constant(batch_dones, dtype=tf.bool)
    )
```

### Important implementation notes
- The replay buffer format does NOT change — it still stores (s, a, r, s', done). The "r" is now an n-step return and "s'" is the state n steps ahead.
- When `done=True`, the bootstrap term `gamma^n * V(s_n)` is zeroed out, which correctly handles partial n-step sequences.
- The n-step buffer per env is a deque with maxlen=n. When full, it auto-drops oldest entries after flush.
- When an episode terminates, ALL remaining entries in that env's buffer must be flushed (computing the appropriate partial return for each).

Actually, let me reconsider the flush logic. When the buffer is full (3 entries), we compute the 3-step return from entries [0,1,2] and add (s_0, a_0, G_3, s_2, d_2). Then the deque auto-drops entry 0 when entry 3 is added next step. But this means we miss transitions starting from entries 1 and 2!

**Corrected approach**: Use a sliding window. Don't clear the buffer when full — let the deque handle it. Only flush explicitly on episode termination.

```python
# Simpler approach:
for i in range(n_envs):
    n_step_buffers[i].append((states_np[i], actions_np[i], rewards_np[i], next_states_np[i], dones_np[i]))

    # Add n-step transition when buffer is full
    if len(n_step_buffers[i]) == agent.n_step:
        G = 0.0
        for j in reversed(range(agent.n_step)):
            _, _, r_j, _, d_j = n_step_buffers[i][j]
            if d_j and j < agent.n_step - 1:
                G = r_j
            else:
                G = r_j + agent.gamma * G
        s_0, a_0, _, _, _ = n_step_buffers[i][0]
        _, _, _, s_n, d_n = n_step_buffers[i][-1]
        batch_states.append(s_0); batch_actions.append(a_0)
        batch_rewards.append(G); batch_next_states.append(s_n)
        batch_dones.append(d_n)
        # Deque will auto-drop oldest on next append

    # On episode end, flush all remaining partial sequences
    if dones_np[i]:
        while len(n_step_buffers[i]) > 0:
            G = 0.0
            for j in reversed(range(len(n_step_buffers[i]))):
                _, _, r_j, _, d_j = n_step_buffers[i][j]
                G = r_j + agent.gamma * G
            s_0, a_0, _, _, _ = n_step_buffers[i][0]
            _, _, _, s_n, d_n = n_step_buffers[i][-1]  # d_n is True
            batch_states.append(s_0); batch_actions.append(a_0)
            batch_rewards.append(G); batch_next_states.append(s_n)
            batch_dones.append(True)
            n_step_buffers[i].popleft()
```

### Expected mechanism
N-step returns reduce bias in Q-targets at the cost of slightly more variance. For CartPole with gamma=0.999, the 1-step bootstrap has high bias because V(s') is initially inaccurate. With 3-step returns, 3 real rewards are used before bootstrapping, providing more accurate targets early in training. This speeds up credit assignment: the agent learns faster that "tilting pole → low future reward" without needing many backup iterations.

### Rollback plan
git revert the merge commit
