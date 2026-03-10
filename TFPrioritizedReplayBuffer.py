import tensorflow as tf
import numpy as np
from typing import Tuple

from sumtree import TFSumTree

class TFPrioritizedReplayBuffer:
    """
    TensorFlow-based Prioritized Experience Replay Buffer using optimized SumTree.
    
    Integrates TFSumTree for efficient proportional sampling with standard replay buffer
    for experience storage. Supports importance sampling with configurable α and β parameters.
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 buffer_size: int = 524_288,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_end: float = 1.0,
                 beta_frames: int = 100_000,
                 epsilon: float = 1e-6):
        """
        Initialize the Prioritized Replay Buffer.
        
        Args:
            state_size: Dimension of state space
            action_size: Dimension of action space  
            buffer_size: Maximum number of experiences to store
            alpha: Priority exponent (0=uniform, 1=full prioritization)
            beta_start: Initial importance sampling exponent
            beta_end: Final importance sampling exponent
            beta_frames: Number of frames to anneal beta from start to end
            epsilon: Small constant to prevent zero priorities
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        # Experience storage using TensorFlow variables
        self.states = tf.Variable(
            tf.zeros((buffer_size, state_size), dtype=tf.float32),
            trainable=False, name="per_states")
        self.actions = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.int32),
            trainable=False, name="per_actions")
        self.rewards = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.float32),
            trainable=False, name="per_rewards")
        self.next_states = tf.Variable(
            tf.zeros((buffer_size, state_size), dtype=tf.float32),
            trainable=False, name="per_next_states")
        self.dones = tf.Variable(
            tf.zeros((buffer_size,), dtype=tf.bool),
            trainable=False, name="per_dones")
        
        # Priority management with SumTree
        self.sumtree = TFSumTree(buffer_size)
        
        # Initialize sumtree data with buffer indices [0, 1, 2, ..., buffer_size-1]
        buffer_indices = np.arange(buffer_size, dtype=np.float32)
        initial_priorities = np.zeros(buffer_size, dtype=np.float32)  # Start with zero priorities
        self.sumtree.batch_add(initial_priorities, buffer_indices)
        self.sumtree.rebuild_tree()
        
        # Buffer state tracking
        self.current_size = 0
        self.current_idx = 0
        self.max_priority = 1.0  # Initial max priority for new experiences
        self.frame_count = 0  # For beta annealing
        
    def _get_beta(self) -> float:
        """Get current beta value with linear annealing."""
        if self.frame_count >= self.beta_frames:
            return self.beta_end
        
        fraction = self.frame_count / self.beta_frames
        return self.beta_start + fraction * (self.beta_end - self.beta_start)
    
    def add(self, states_batch: tf.Tensor, actions_batch: tf.Tensor, 
            rewards_batch: tf.Tensor, next_states_batch: tf.Tensor, 
            dones_batch: tf.Tensor) -> None:
        """
        Add a batch of experiences to the buffer with maximum priority.
        
        Args:
            states_batch: Batch of states [batch_size, state_size]
            actions_batch: Batch of actions [batch_size]
            rewards_batch: Batch of rewards [batch_size]
            next_states_batch: Batch of next states [batch_size, state_size]
            dones_batch: Batch of done flags [batch_size]
        """
        batch_size = tf.shape(states_batch)[0].numpy()
        
        # Calculate indices for this batch - simple slice assignment
        start_idx = self.current_idx
        end_idx = start_idx + batch_size
        
        # Assign experiences to buffer (no wraparound needed with our constraints)
        idx_slice = slice(start_idx, end_idx)
        self.states[idx_slice].assign(states_batch)
        self.actions[idx_slice].assign(actions_batch)
        self.rewards[idx_slice].assign(rewards_batch)
        self.next_states[idx_slice].assign(next_states_batch)
        self.dones[idx_slice].assign(dones_batch)
        
        # Assign max priority to new experiences
        new_priorities = np.full(batch_size, self.max_priority ** self.alpha, dtype=np.float32)
        
        # Update priorities in sumtree
        self.sumtree.priorities[start_idx:end_idx] = new_priorities
        self.sumtree.rebuild_tree(start_leaf=start_idx, end_leaf=end_idx)
        
        # Update buffer state
        self.current_idx = (self.current_idx + batch_size) % self.buffer_size
        self.current_size = min(self.current_size + batch_size, self.buffer_size)
        self.frame_count += batch_size
    
    def sample(self, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, 
                                               tf.Tensor, tf.Tensor, tf.Tensor, np.ndarray]:
        """
        Sample a batch of experiences using prioritized sampling.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, buffer_indices)
            where weights are importance sampling weights and buffer_indices are for priority updates
        """
        if self.current_size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Generate random sampling points
        total_priority = self.sumtree.total
        if total_priority <= 0:
            # Fallback to uniform sampling if no priorities
            random_points = np.random.uniform(0, self.epsilon * self.current_size, batch_size)
        else:
            random_points = np.random.uniform(0, total_priority, batch_size)
        
        # Use sumtree to find corresponding buffer indices
        # batch_get returns (data_indices, priorities, data_vals)
        buffer_indices, priorities, data_vals = self.sumtree.batch_get(random_points)
        
        # Convert to numpy for indexing
        buffer_indices = buffer_indices.numpy().astype(np.int32)
        priorities = priorities.numpy()
        
        # Ensure indices are within current buffer size and valid
        buffer_indices = np.clip(buffer_indices, 0, self.current_size - 1)
        
        # Gather experiences using buffer indices
        states = tf.gather(self.states, buffer_indices)
        actions = tf.gather(self.actions, buffer_indices)
        rewards = tf.gather(self.rewards, buffer_indices)
        next_states = tf.gather(self.next_states, buffer_indices)
        dones = tf.gather(self.dones, buffer_indices)
        
        # Calculate importance sampling weights
        # P(i) = priority_i^α / Σ(priority_j^α)
        # weight_i = (1/N * 1/P(i))^β
        sampling_probabilities = priorities / total_priority
        sampling_probabilities = np.maximum(sampling_probabilities, self.epsilon)  # Avoid division by zero
        
        self.beta = self._get_beta()
        weights = (1.0 / self.current_size / sampling_probabilities) ** self.beta
        
        # Normalize weights by max weight for stability
        max_weight = np.max(weights)
        weights = weights / max_weight
        weights = tf.constant(weights, dtype=tf.float32)
        
        return states, actions, rewards, next_states, dones, weights, buffer_indices
    
    def update_priorities(self, buffer_indices: np.ndarray, td_errors: tf.Tensor) -> None:
        """
        Update priorities for sampled experiences based on TD errors.
        
        Args:
            buffer_indices: Buffer indices corresponding to sampled experiences
            td_errors: TD errors for the sampled experiences [batch_size]
        """
        td_errors = td_errors.numpy()
        buffer_indices = buffer_indices.astype(np.int32)
        
        # Convert TD errors to priorities: priority = |TD_error| + ε
        raw_priorities = np.abs(td_errors) + self.epsilon
        
        # Update max priority for future new experiences (before alpha scaling)
        self.max_priority = max(self.max_priority, np.max(raw_priorities))
        
        # Apply alpha exponent for sumtree storage
        new_priorities = raw_priorities ** self.alpha
        
        # Update priorities in sumtree
        # Since sumtree.data[i] = i, buffer index directly maps to sumtree leaf position
        for i, buffer_idx in enumerate(buffer_indices):
            if 0 <= buffer_idx < self.buffer_size:
                self.sumtree.priorities[buffer_idx] = new_priorities[i]
        
        # Rebuild affected parts of tree efficiently
        if len(buffer_indices) > 0:
            # Find the range of affected indices for efficient partial rebuild
            valid_indices = buffer_indices[(buffer_indices >= 0) & (buffer_indices < self.buffer_size)]
            if len(valid_indices) > 0:
                min_idx = int(np.min(valid_indices))
                max_idx = int(np.max(valid_indices)) + 1
                self.sumtree.rebuild_tree(start_leaf=min_idx, end_leaf=max_idx)
    
    def get_stats(self) -> dict:
        """Get buffer statistics for monitoring."""
        return {
            'size': self.current_size,
            'capacity': self.buffer_size,
            'current_idx': self.current_idx,
            'max_priority': self.max_priority,
            'total_priority': self.sumtree.total,
            'beta': self._get_beta(),
            'frame_count': self.frame_count,
            'alpha': self.alpha
        }
    
    def __repr__(self) -> str:
        """String representation of the buffer."""
        stats = self.get_stats()
        return (f"TFPrioritizedReplayBuffer("
                f"size={stats['size']}/{stats['capacity']}, "
                f"max_priority={stats['max_priority']:.4f}, "
                f"total_priority={stats['total_priority']:.2f}, "
                f"beta={stats['beta']:.3f})")
