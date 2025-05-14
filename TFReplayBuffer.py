import tensorflow as tf

class TFReplayBuffer:
    def __init__(self, state_size, action_size, buffer_size=50_000):
        # Preallocate tensors on CPU (faster transfers later)
        self.states = tf.Variable(
            tf.zeros((buffer_size, state_size), dtype=tf.float32),
            trainable=False, name="replay_states")
        self.actions = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.int32),
            trainable=False, name="replay_actions")
        self.rewards = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.float32),
            trainable=False, name="replay_rewards")
        self.next_states = tf.Variable(
            tf.zeros((buffer_size, state_size), dtype=tf.float32),
            trainable=False, name="replay_next_states")
        self.dones = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.bool),
            trainable=False, name="replay_dones")
        
        self.buffer_size = buffer_size
        self.current_size = 0
        self.current_idx = 0
    
    def add(self, states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch):
        # Add batch of experiences (vectorized add for parallel environments)
        batch_size = states_batch.shape[0]
        
        # Handle wrapping around buffer end
        indices = tf.range(self.current_idx, self.current_idx + batch_size) % self.buffer_size
        
        # Use scatter updates to efficiently update buffer
        self.states.scatter_update(tf.IndexedSlices(states_batch, indices))
        self.actions.scatter_update(tf.IndexedSlices(actions_batch, indices))
        self.rewards.scatter_update(tf.IndexedSlices(rewards_batch, indices))
        self.next_states.scatter_update(tf.IndexedSlices(next_states_batch, indices))
        self.dones.scatter_update(tf.IndexedSlices(dones_batch, indices))
        
        # Update tracking variables
        self.current_idx = (self.current_idx + batch_size) % self.buffer_size
        self.current_size = min(self.current_size + batch_size, self.buffer_size)
    
    def sample(self, batch_size):
        # Sample a batch of experiences
        indices = tf.random.uniform(
            [batch_size], minval=0, maxval=self.current_size, dtype=tf.int32)
        
        # Gather sampled experiences - transfer to GPU happens here if needed
        states = tf.gather(self.states[:self.current_size], indices)
        actions = tf.gather(self.actions[:self.current_size], indices)
        rewards = tf.gather(self.rewards[:self.current_size], indices)
        next_states = tf.gather(self.next_states[:self.current_size], indices)
        dones = tf.gather(self.dones[:self.current_size], indices)
        
        return states, actions, rewards, next_states, dones