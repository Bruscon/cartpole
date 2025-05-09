import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import TensorBoard
from collections import deque
import random
import time
import os

verbose = 1

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#use mixed precision to encourage GPU usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

#allows GPU memory to be allocated as needed
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

# Check for GPU
if verbose > 1:
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    print("Using GPU:", tf.config.list_physical_devices('GPU'))
    print("CUDA built:", tf.test.is_built_with_cuda())

#setup logging
log_dir = f"logs/dqn_{time.strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Create log directory for TensorBoard
current_time = time.strftime('%Y%m%d-%H%M%S')
log_dir = f"logs/dqn_{current_time}"
os.makedirs(log_dir, exist_ok=True)

# Create summary writer for TensorBoard once
summary_writer = tf.summary.create_file_writer(log_dir)

# Global counter for tracking total steps
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)


class DQNAgent:
    def __init__(self, state_size, action_size, summary_writer, global_step):
        self.state_size = state_size
        self.action_size = action_size
        self.summary_writer = summary_writer
        self.global_step = global_step
        
        # Hyperparameters
        self.memory = deque(maxlen=20000)  # Experience replay buffer - increased size
        self.gamma = 0.9                   # Discount factor
        self.epsilon = 1.0                 # Exploration rate
        self.epsilon_min = 0.01            # Minimum exploration probability
        self.epsilon_decay = 0.995         # Exponential decay rate for exploration
        self.learning_rate = 0.001         # Learning rate
        self.batch_size = 128              # Size of batches for training
        self.train_start = 1000            # Minimum experiences before training
        self.update_target_frequency = 10  # How often to update target network (episodes)
        
        # Main model - trained every step
        self.model = self._build_model()
        
        # Target model - used for more stable Q-value predictions
        self.target_model = self._build_model()
        
        # Initialize target model with same weights as main model
        self.update_target_model()
        
        # Track metrics for TensorBoard
        self.episode_loss_history = []
    
    def _build_model(self):
        """Neural Network for Deep-Q learning Model"""
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values for all actions in current state
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the network using randomly sampled experiences"""
        # Check if we have enough experiences
        if len(self.memory) < self.train_start:
            return 0.0  # Return 0 loss if no training performed
        
        # Sample a random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batches
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Calculate target Q values
        targets = self.model.predict(states, verbose=0)
        
        # Calculate target Q value for action using target model
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Double DQN: use main model to select action, target model to get Q-value
        max_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        
        # Update target for actions taken
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # DDQN update
                targets[i][actions[i]] = rewards[i] + self.gamma * target_q_values[i][max_actions[i]]
        
        # Train the network - without callbacks
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Increment global step and track loss
        loss = history.history['loss'][0]
        self.episode_loss_history.append(loss)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def log_episode_stats(self, episode, score, avg_score, avg_reward):
        """Log episode statistics to TensorBoard"""
        # Calculate average loss for this episode
        avg_loss = np.mean(self.episode_loss_history) if self.episode_loss_history else 0.0
        
        # Log scalar metrics
        with self.summary_writer.as_default():
            tf.summary.scalar('score', score, step=episode)
            tf.summary.scalar('average_score_100', avg_score, step=episode)
            tf.summary.scalar('average_reward', avg_reward, step=episode)
            tf.summary.scalar('average_loss', avg_loss, step=episode)
            tf.summary.scalar('epsilon', self.epsilon, step=episode)
            
            # Log model weights and gradients every 10 episodes
            if episode % 10 == 0:
                for i, layer in enumerate(self.model.layers):
                    for j, weight in enumerate(layer.weights):
                        tf.summary.histogram(f"layer_{i}_weight_{j}", weight, step=episode)
            
            # Ensure metrics are written to disk
            self.summary_writer.flush()
            
        # Reset loss history for next episode
        self.episode_loss_history = []


def main():
    MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
    MAX_VELOCITY = 2.0  # Reasonable maximum cart velocity threshold

    # Create environments - one for training, one for rendering
    env = gym.make('CartPole-v1')
    render_env = gym.make('CartPole-v1', render_mode='human')
    
    # Get environment information
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent with the TensorBoard writer
    agent = DQNAgent(state_size, action_size, summary_writer, global_step)
    
    # Training parameters
    EPISODES = 500
    
    # Keep track of rewards per episode
    scores = []
    avg_scores = []
    
    for e in range(EPISODES):
        # Determine if we should render this episode
        if e % 10 == 0:
            current_env = render_env
        else:
            current_env = env
        
        # Reset environment
        state, _ = current_env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        episode_rewards = []  # Track all rewards for episode statistics
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, env_reward, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
            
            # Extract values for custom reward
            cart_position = next_state[0]  # Not used directly in reward
            cart_velocity = next_state[1]  # Used for velocity component
            pole_angle = next_state[2]     # Used for angle component
            pole_velocity = next_state[3]  # Not used directly in reward
            
            # Calculate custom reward components
            angle_reward = 1.0 - (abs(pole_angle) / MAX_ANGLE)  # 1.0 when vertical, 0.0 at maximum angle
            velocity_reward = max(0, 1.0 - (abs(cart_velocity) / MAX_VELOCITY))  # 1.0 when stationary
            
            # Combined reward (weighted 70% angle, 30% velocity)
            custom_reward = 0.7 * angle_reward + 0.3 * velocity_reward
            
            # Apply termination penalty if the episode ended early (not due to max steps)
            if done and score < 499:
                custom_reward = -10  # Maintain the same severe penalty for early termination
            
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience with custom reward
            agent.remember(state, action, custom_reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Track rewards
            episode_rewards.append(custom_reward)
            score += 1
            
            # Train on past experiences (replay)
            agent.replay()
            
            # Increment global step
            global_step.assign_add(1)
            
            if done:
                break
        
        # Update target model periodically
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()
            print(f"Episode: {e}/{EPISODES}, Target model updated")
        
        # Save scores
        scores.append(score)
        
        # Calculate average score of last 100 episodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        # Calculate average reward per step for this episode
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
        
        # Log episode statistics to TensorBoard
        agent.log_episode_stats(e, score, avg_score, avg_reward)
        
        print(f"Episode: {e}/{EPISODES}, Score: {score}, Avg Score: {avg_score:.2f}, " +
              f"Avg Reward: {avg_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
        
        # If we've solved the environment (avg score >= 195 over 100 episodes), stop
        if len(scores) >= 100 and avg_score >= 195:
            print(f"Environment solved in {e} episodes! Average score: {avg_score:.2f}")
            break
    
    agent.model.save('trained_cartpole_model.keras')

    # Close environments
    env.close()
    render_env.close()

if __name__ == "__main__":
    main()