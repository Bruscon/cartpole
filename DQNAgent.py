import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
from collections import deque
import os
import random

from TFReplayBuffer import TFReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, log_dir, initial_model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.log_dir = log_dir
        self.initial_model_path = initial_model_path
        
        # Hyperparameters
        self.memory_len = 50_000            # Experience replay buffer
        self.gamma = 0.99                   # Discount factor
        self.epsilon = 1.0                  # Exploration rate
        self.epsilon_min = 0.005             # Minimum exploration probability
        self.epsilon_decay = 0.99            # Exponential decay rate for exploration
        self.batch_size = 2048              # Size of batches for training
        self.learning_rate = .01             # Initial learning rate
        self.learning_rate_decay = .999       # learning rate decay 
        self.epochs = 10
        self.train_frequency = 1           # How many time steps between training runs
        self.update_target_frequency = 100  # How often to HARD update target network (steps)
        self.tau = 0.05                    # Soft update parameter (happens every training)

        self.train_start = 2* self.batch_size  # Minimum experiences before training

        # create memory object
        self.memory = TFReplayBuffer(state_size, action_size, buffer_size=self.memory_len)
        
        # learning rate scheduler for adam optimizer
        self.lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=1,
            decay_rate=self.learning_rate_decay)
        self.optimizer_steps = 0

        # optimizer config 
        self.optimizer = Adam(learning_rate=self.lr_schedule, clipnorm=1.0)
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            # Wrap optimizer for mixed precision
            self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)

        # Load or build the main model
        if initial_model_path and os.path.exists(initial_model_path):
            print(f"Loading initial model from: {initial_model_path}")
            self.model = self._build_model()  # Create a fresh model with our architecture
            # Only load weights, not the optimizer state
            temp_model = tf.keras.models.load_model(initial_model_path)
            self.model.set_weights(temp_model.get_weights())
            del temp_model  # Free memory
        else:
            self.model = self._build_model()

        
        # Target model - used for more stable Q-value predictions
        self.target_model = self._build_model()
        
        # Initialize target model with same weights as main model
        self.update_target_model()
        
    
    def _build_model(self):
        """Neural Network for Deep-Q learning Model"""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(24, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss=Huber(delta=10.0), optimizer=self.optimizer)
        # model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def update_target_model(self, tau=1.0):
        """
        Update target model weights.
        
        Args:
            tau: Float between 0 and 1. If tau=1.0, performs a hard update 
                 (complete copy). If tau<1.0, performs a soft update where
                 target_weights = tau * model_weights + (1 - tau) * target_weights
        """
        if tau >= 1.0:
            # Standard hard update (complete copy)
            self.target_model.set_weights(self.model.get_weights())
        else:
            # Soft update
            model_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            
            # Blend weights
            updated_weights = []
            for model_w, target_w in zip(model_weights, target_weights):
                updated_w = tau * model_w + (1 - tau) * target_w
                updated_weights.append(updated_w)
            
            self.target_model.set_weights(updated_weights)

    def get_current_learning_rate(self):
        return self.lr_schedule(self.optimizer_steps).numpy()
    
    # Keep the NumPy version for compatibility with single environment training
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    @tf.function
    def _process_batch(self, states, actions, rewards, next_states, dones):
        """TF function for any preprocessing needed before memory storage"""
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        dones = tf.cast(dones, tf.bool)
        return states, actions, rewards, next_states, dones

    def remember_batch(self, states, actions, rewards, next_states, dones):
        """Store batch of experiences in memory - no @tf.function here"""
        # Process in graph mode, then store in eager mode
        processed = self._process_batch(states, actions, rewards, next_states, dones)
        self.memory.add(*processed)
    
    @tf.function
    def get_greedy_actions(self, states_batch):
        """Only computes the greedy actions using the model - no randomness here"""
        q_values = self.model(states_batch, training=False)
        return tf.argmax(q_values, axis=1, output_type=tf.int32)

    # This function handles all randomness outside the graph. For some reason the decorator breaks it. Something about randomness in tf graphs being weird
    def explore_batch(self, actions, seed=None):
        """Handles epsilon-greedy exploration using pure TensorFlow operations"""
        batch_size = tf.shape(actions)[0]
        
        # Set seed for reproducibility if provided
        if seed is not None:
            tf.random.set_seed(seed)
        
        # Generate exploration mask using TensorFlow random ops
        explore_mask = tf.random.uniform(shape=[batch_size], minval=0, maxval=1) < self.epsilon
        
        # Generate random actions for the entire batch
        random_actions = tf.random.uniform(
            shape=[batch_size], 
            minval=0, 
            maxval=self.action_size, 
            dtype=tf.int32
        )
        
        # Use tf.where to select either the original action or random action based on explore_mask
        final_actions = tf.where(
            condition=explore_mask,
            x=random_actions,
            y=actions
        )
        
        return final_actions

    @tf.function
    def predict_batch(self, states):
        """Direct model prediction without .predict() to avoid unnecessary overhead"""
        return self.model(states)
    
    @tf.function(reduce_retracing=True)
    def _compute_targets(self, states, actions, rewards, next_states, dones):
        """Compute Q-value targets using TensorFlow operations"""
        # Define the compute dtype based on global policy
        compute_dtype = tf.float32
        
        # Convert inputs to tensors with explicit casting
        states = tf.cast(tf.convert_to_tensor(states), compute_dtype)
        next_states = tf.cast(tf.convert_to_tensor(next_states), compute_dtype)
        rewards = tf.cast(tf.convert_to_tensor(rewards), compute_dtype)
        actions = tf.cast(tf.convert_to_tensor(actions), tf.int32)
        dones = tf.cast(tf.convert_to_tensor(dones), tf.bool)
        
        # Get current Q values
        current_q = tf.cast(self.model(states), compute_dtype)
        
        # Get next Q values from target network
        next_q_values = tf.cast(self.target_model(next_states), compute_dtype)
        
        # For DDQN, get actions from main network
        model_next_q = tf.cast(self.model(next_states), compute_dtype)
        next_actions = tf.cast(tf.argmax(model_next_q, axis=1), tf.int32)
        
        # Create indices for gathering values
        batch_size = tf.shape(states)[0]
        batch_indices = tf.range(batch_size, dtype=tf.int32)
        action_indices = tf.stack([batch_indices, actions], axis=1)
        next_action_indices = tf.stack([batch_indices, next_actions], axis=1)
        
        # Gather the Q-values for the actions taken
        q_values = tf.gather_nd(current_q, action_indices)
        
        # Gather the Q-values for the next best actions
        next_q = tf.gather_nd(next_q_values, next_action_indices)
        
        # Cast all variables to compute_dtype to ensure compatibility
        gamma = tf.cast(self.gamma, compute_dtype)
        next_q = tf.cast(next_q, compute_dtype)
        rewards = tf.cast(rewards, compute_dtype)
        
        # Create the targets
        # For terminal states, target is just the reward
        # For non-terminal states, target is reward + gamma * next_q
        done_mask = tf.cast(dones, compute_dtype)
        targets = rewards + (1.0 - done_mask) * gamma * next_q
        
        # Create a copy of current_q for updating specific actions
        updated_q_values = current_q
        
        # Update only the relevant action values using scatter_nd
        indices = action_indices
        updates = targets
        updated_q_values = tf.tensor_scatter_nd_update(updated_q_values, indices, updates)
        
        return updated_q_values

    def replay(self):
        """Train the network using randomly sampled experiences with vectorized operations"""
        # Check if we have enough experiences
        if self.memory.current_size < self.train_start:
            return 0.0 
        
        # Sample a random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Use TensorFlow to compute target Q-values
        targets = self._compute_targets(states, actions, rewards, next_states, dones)
        
        # Train the network with optimized batch processing
        history = self.model.fit(
            states, 
            targets, 
            epochs=self.epochs, 
            verbose=0, 
            batch_size=self.batch_size
        )
        
        # decay epsilon 
        if (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

        # This decays the learning rate
        self.optimizer_steps += 1

        # Increment global step and track loss
        loss = history.history['loss'][0]
            
        return loss
    
    def save_model(self, episode):
        """Save model checkpoint"""
        model_path = os.path.join(self.log_dir, f"model_episode_{episode}.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")