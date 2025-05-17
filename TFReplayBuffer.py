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
        # Get the range of indices to update
        batch_size = tf.shape(states_batch)[0]
        end_idx = self.current_idx + batch_size
        
        # Handle the case where the batch wraps around the buffer
        if end_idx <= self.buffer_size:
            # Simple case: batch fits without wrapping
            idx_slice = slice(self.current_idx, end_idx)
            self.states[idx_slice].assign(states_batch)
            self.actions[idx_slice].assign(actions_batch)
            self.rewards[idx_slice].assign(rewards_batch)
            self.next_states[idx_slice].assign(next_states_batch)
            self.dones[idx_slice].assign(dones_batch)
        else:
            # Complex case: batch wraps around
            first_size = self.buffer_size - self.current_idx
            second_size = batch_size - first_size
            
            # First part (up to buffer end)
            first_slice = slice(self.current_idx, self.buffer_size)
            self.states[first_slice].assign(states_batch[:first_size])
            self.actions[first_slice].assign(actions_batch[:first_size])
            self.rewards[first_slice].assign(rewards_batch[:first_size])
            self.next_states[first_slice].assign(next_states_batch[:first_size])
            self.dones[first_slice].assign(dones_batch[:first_size])
            
            # Second part (wrapping to buffer start)
            second_slice = slice(0, second_size)
            self.states[second_slice].assign(states_batch[first_size:])
            self.actions[second_slice].assign(actions_batch[first_size:])
            self.rewards[second_slice].assign(rewards_batch[first_size:])
            self.next_states[second_slice].assign(next_states_batch[first_size:])
            self.dones[second_slice].assign(dones_batch[first_size:])
        
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