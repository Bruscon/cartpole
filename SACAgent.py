import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, clone_model
from tensorflow.keras.layers import Activation, Dense, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from collections import deque
import os
import random

from TFPrioritizedReplayBuffer import TFPrioritizedReplayBuffer


class DuelingCombine(keras.layers.Layer):
    """Combines value and advantage streams: Q = V + A - mean(A)"""
    def call(self, inputs):
        value, advantage = inputs
        return value + advantage - keras.ops.mean(advantage, axis=1, keepdims=True)


class SACAgent:
    def __init__(self, state_size, action_size, log_dir, initial_model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.log_dir = log_dir
        self.initial_model_path = initial_model_path
        
        # Hyperparameters (mostly inherited from DQN)
        self.clipnorm = 2.0                 # gradient clipping
        self.memory_len = 2**17             # Experience replay buffer
        self.gamma = 0.999                  # Discount factor
        self.n_step = 8                     # Multi-step return horizon
        self.batch_size = 256               # Size of batches for training
        self.learning_rate = 0.03          # Initial learning rate
        self.learning_rate_decay = 0.999    # learning rate decay
        self.epochs = 3                     # number of training loops per step
        self.train_frequency = 1            # How many time steps between training runs
        self.update_target_frequency = 1000000  # How often to HARD update target network (steps). effectively disabled
        self.tau = 0.1                      # Soft update parameter
        
        # SAC specific hyperparameters
        self.target_entropy = -1.2*float(action_size)  # Target entropy = -dim(A)
        self.log_alpha = tf.Variable(0.0, trainable=True)  # Entropy temperature (log scale for stability)
        
        # Prioritized replay hyperparameters (same as DQN)
        self.priority_alpha = 0.7                    # prioritization parameter
        self.beta_start = 0.3               # reduces bias
        self.beta_end = 1.0                 # reduces bias
        self.beta_frames = 50_000           # reduces bias
        
        self.train_start = 2 * self.batch_size  # Minimum experiences before training
        
        # Create memory object (reuse existing)
        self.memory = TFPrioritizedReplayBuffer(
            state_size=state_size,
            action_size=action_size, 
            buffer_size=self.memory_len,
            alpha=self.priority_alpha,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            beta_frames=self.beta_frames
        )
        
        # Learning rate scheduler
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1,
            decay_rate=self.learning_rate_decay)
        self.optimizer_steps = 0
        
        # Create optimizers for each network
        self.q1_optimizer = Adam(learning_rate=self.lr_schedule, clipnorm=self.clipnorm)
        self.q2_optimizer = Adam(learning_rate=self.lr_schedule, clipnorm=self.clipnorm)
        self.policy_optimizer = Adam(learning_rate=self.lr_schedule, clipnorm=self.clipnorm)
        self.alpha_optimizer = Adam(learning_rate=3e-4)  # Fixed LR for temperature
        
        # Handle mixed precision if enabled
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.q1_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.q1_optimizer)
            self.q2_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.q2_optimizer)
            self.policy_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.policy_optimizer)
        
        # Build networks
        self.q1_model = self._build_q_model()
        self.q2_model = self._build_q_model()
        self.policy_model = self._build_policy_model()
        
        # Target networks
        self.q1_target = self._build_q_model()
        self.q2_target = self._build_q_model()
        
        # Initialize target networks
        self.update_target_models(tau=1.0)
        
    def _build_q_model(self):
        """Dueling Q-Network architecture"""
        ortho_relu = tf.keras.initializers.Orthogonal(gain=1.4142135)
        ortho_lin = tf.keras.initializers.Orthogonal(gain=1.0)
        inputs = Input(shape=(self.state_size,))
        x = Dense(64, activation=None, kernel_initializer=ortho_relu)(inputs)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(64, activation=None, kernel_initializer=ortho_relu)(x)
        x = LayerNormalization()(x)
        x = Activation('relu')(x)

        value = Dense(1, activation='linear', kernel_initializer=ortho_lin)(x)
        advantage = Dense(self.action_size, activation='linear', kernel_initializer=ortho_lin)(x)
        q_values = DuelingCombine()([value, advantage])

        model = Model(inputs=inputs, outputs=q_values)
        return model

    def _build_policy_model(self):
        """Policy Network for discrete actions"""
        ortho_relu = tf.keras.initializers.Orthogonal(gain=1.4142135)
        ortho_lin = tf.keras.initializers.Orthogonal(gain=0.01)
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation=None, kernel_initializer=ortho_relu),
            LayerNormalization(),
            Activation('relu'),
            Dense(64, activation=None, kernel_initializer=ortho_relu),
            LayerNormalization(),
            Activation('relu'),
            Dense(self.action_size, activation='linear', kernel_initializer=ortho_lin)
        ])
        return model
    
    def update_target_models(self, tau=None):
        """Update target networks using Polyak averaging"""
        if tau is None:
            tau = self.tau
            
        # Update Q1 target
        q1_weights = self.q1_model.get_weights()
        q1_target_weights = self.q1_target.get_weights()
        for i in range(len(q1_weights)):
            q1_target_weights[i] = tau * q1_weights[i] + (1 - tau) * q1_target_weights[i]
        self.q1_target.set_weights(q1_target_weights)
        
        # Update Q2 target
        q2_weights = self.q2_model.get_weights()
        q2_target_weights = self.q2_target.get_weights()
        for i in range(len(q2_weights)):
            q2_target_weights[i] = tau * q2_weights[i] + (1 - tau) * q2_target_weights[i]
        self.q2_target.set_weights(q2_target_weights)
    
    def get_current_learning_rate(self):
        return self.lr_schedule(self.optimizer_steps).numpy()
    
    @property
    def alpha(self):
        """Get current temperature value"""
        return tf.exp(self.log_alpha).numpy()
    
    @property
    def epsilon(self):
        """Compatibility property - SAC doesn't use epsilon"""
        return 0.0
    
    @tf.function
    def get_actions(self, states):
       """Sample actions from policy """
       # Get action logits from policy network
       logits = self.policy_model(states, training=False)
       
       # Sample from categorical distribution
       actions = tf.random.categorical(logits, 1, dtype=tf.int32)
       return tf.squeeze(actions, axis=1, name='actions')

    @tf.function
    def _compute_policy_loss(self, states):
        """Compute policy loss for discrete SAC"""
        # Get action probabilities
        logits = self.policy_model(states)
        log_probs = tf.nn.log_softmax(logits)
        probs = tf.nn.softmax(logits)
        
        # Get Q-values for all actions
        q1_values = self.q1_model(states)
        q2_values = self.q2_model(states)
        min_q_values = tf.minimum(q1_values, q2_values)
        
        # Compute policy loss: E[π(a|s) * (α * log π(a|s) - Q(s,a))]
        alpha = tf.exp(self.log_alpha)
        policy_loss = tf.reduce_mean(
            tf.reduce_sum(probs * (alpha * log_probs - min_q_values), axis=1)
        )
        
        # Also return entropy for temperature tuning
        entropy = -tf.reduce_mean(tf.reduce_sum(probs * log_probs, axis=1))
        
        return policy_loss, entropy, probs
    
    @tf.function
    def _compute_q_targets(self, rewards, next_states, dones):
        """Compute soft Q-targets for discrete SAC"""
        # Get next action probabilities
        next_logits = self.policy_model(next_states)
        next_log_probs = tf.nn.log_softmax(next_logits)
        next_probs = tf.nn.softmax(next_logits)
        
        # Get next Q-values from target networks
        next_q1 = self.q1_target(next_states)
        next_q2 = self.q2_target(next_states)
        next_min_q = tf.minimum(next_q1, next_q2)
        
        # Compute soft value: V(s) = E[Q(s,a) - α * log π(a|s)]
        alpha = tf.exp(self.log_alpha)
        next_values = tf.reduce_sum(
            next_probs * (next_min_q - alpha * next_log_probs), axis=1
        )
        
        # Compute targets
        dones_float = tf.cast(dones, tf.float32)
        targets = rewards + (1.0 - dones_float) * (self.gamma ** self.n_step) * next_values
        
        return targets
    
    @tf.function
    def _train_step(self, states, actions, rewards, next_states, dones, is_weights):
        """Single training step for all networks"""
        # Prepare action indices for gathering
        batch_size = tf.shape(states)[0]
        indices = tf.range(batch_size)
        action_indices = tf.stack([indices, tf.cast(actions, tf.int32)], axis=1)
        
        # Train Q-networks
        targets = self._compute_q_targets(rewards, next_states, dones)
        targets = tf.stop_gradient(targets)  # Critical for stability
        
        # Q1 update
        with tf.GradientTape() as tape:
            q1_values = self.q1_model(states, training=True)
            q1_selected = tf.gather_nd(q1_values, action_indices)
            
            # Huber loss with importance weights
            huber = tf.keras.losses.Huber(delta=1.0, reduction='none')
            q1_loss = huber(targets, q1_selected)
            q1_loss = tf.reduce_mean(q1_loss * is_weights)
            
            if hasattr(self.q1_optimizer, 'get_scaled_loss'):
                q1_loss = self.q1_optimizer.get_scaled_loss(q1_loss)
        
        q1_grads = tape.gradient(q1_loss, self.q1_model.trainable_variables)
        if hasattr(self.q1_optimizer, 'get_unscaled_gradients'):
            q1_grads = self.q1_optimizer.get_unscaled_gradients(q1_grads)
        self.q1_optimizer.apply_gradients(zip(q1_grads, self.q1_model.trainable_variables))
        
        # Q2 update (similar)
        with tf.GradientTape() as tape:
            q2_values = self.q2_model(states, training=True)
            q2_selected = tf.gather_nd(q2_values, action_indices)
            
            q2_loss = huber(targets, q2_selected)
            q2_loss = tf.reduce_mean(q2_loss * is_weights)
            
            if hasattr(self.q2_optimizer, 'get_scaled_loss'):
                q2_loss = self.q2_optimizer.get_scaled_loss(q2_loss)
        
        q2_grads = tape.gradient(q2_loss, self.q2_model.trainable_variables)
        if hasattr(self.q2_optimizer, 'get_unscaled_gradients'):
            q2_grads = self.q2_optimizer.get_unscaled_gradients(q2_grads)
        self.q2_optimizer.apply_gradients(zip(q2_grads, self.q2_model.trainable_variables))
        
        # Policy update
        with tf.GradientTape() as tape:
            policy_loss, entropy, _ = self._compute_policy_loss(states)
            
            if hasattr(self.policy_optimizer, 'get_scaled_loss'):
                policy_loss = self.policy_optimizer.get_scaled_loss(policy_loss)
        
        policy_grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
        if hasattr(self.policy_optimizer, 'get_unscaled_gradients'):
            policy_grads = self.policy_optimizer.get_unscaled_gradients(policy_grads)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))
        
        # Temperature update
        with tf.GradientTape() as tape:
            alpha = tf.exp(self.log_alpha)
            alpha_loss = -alpha * (entropy + self.target_entropy)
        
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        # Compute TD errors for prioritization (use Q1)
        td_errors = targets - q1_selected
        
        # Return metrics
        avg_q_loss = (q1_loss + q2_loss) / 2.0
        return avg_q_loss, td_errors, entropy
    
    def replay(self):
        """Train the networks using prioritized experience replay"""
        if self.memory.current_size < self.train_start:
            return 0.0, 0.0

        # Sample batch
        states, actions, rewards, next_states, dones, is_weights, indices = self.memory.sample(self.batch_size)

        # Run training step
        loss, td_errors, entropy = self._train_step(states, actions, rewards, next_states, dones, is_weights)

        # Update priorities
        self.memory.update_priorities(indices, td_errors)

        # Soft update target networks
        self.update_target_models()

        # Increment optimizer steps
        self.optimizer_steps += 1

        return loss.numpy(), tf.reduce_mean(tf.abs(td_errors)).numpy()

    @tf.function
    def get_deterministic_actions(self, states):
        """Get most likely actions for evaluation"""
        logits = self.policy_model(states, training=False)
        return tf.argmax(logits, axis=1)
    
    def save_model(self, episode):
        """Save all models"""
        base_path = os.path.join(self.log_dir, f"model_episode_{episode}")
        
        # Save Q-networks
        self.q1_model.save(f"{base_path}_q1.keras")
        self.q2_model.save(f"{base_path}_q2.keras")
        
        # Save policy
        self.policy_model.save(f"{base_path}_policy.keras")
        
        # Save temperature
        np.save(f"{base_path}_log_alpha.npy", self.log_alpha.numpy())
    
    def load_model(self, base_path):
        """Load all models"""
        self.q1_model = tf.keras.models.load_model(f"{base_path}_q1.keras")
        self.q2_model = tf.keras.models.load_model(f"{base_path}_q2.keras")
        self.policy_model = tf.keras.models.load_model(f"{base_path}_policy.keras")
        
        # Load temperature
        self.log_alpha.assign(np.load(f"{base_path}_log_alpha.npy"))
        
        # Update targets
        self.update_target_models(tau=1.0)
