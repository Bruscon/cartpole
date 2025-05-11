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
import shutil
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
import argparse

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='DQN training for CartPole')
parser.add_argument('--log-dir', type=str, help='Directory for logs')
parser.add_argument('--model', type=str, help='Path to initial model')
args = parser.parse_args()

# Use the log directory if provided, otherwise create one
if args.log_dir:
    log_dir = args.log_dir
else:
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_dir = f"logs/dqn_{current_time}"

# load the initial model from the arguments
if args.model:
    INITIAL_MODEL_PATH = args.model
else:
    INITIAL_MODEL_PATH = None

os.makedirs(log_dir, exist_ok=True)

# Archive a copy of this script to the logging folder 
script_path = os.path.abspath(__file__)
script_name = os.path.basename(script_path)
script_archive_path = os.path.join(log_dir, script_name)
shutil.copy2(script_path, script_archive_path)
print(f"Script archived to: {script_archive_path}")

# Global counter for tracking total steps
global_step = tf.Variable(0, trainable=False, dtype=tf.int64)

# Create TensorBoard writer
summary_writer = tf.summary.create_file_writer(log_dir)


class CartPoleCenteringWrapper(ObservationWrapper):
    """
    Wrapper that adds normalized distance from cart center as an additional observation.
    This helps the agent learn to keep the cart centered.
    """
    def __init__(self, env):
        super().__init__(env)
        # Update observation space to include normalized distance from center
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(old_shape[0],),  # Keep same dimension as we'll normalize the position
            dtype=np.float32
        )
        
        # Track max position for normalization
        self.max_position = 2.4  # CartPole max position
    
    def observation(self, observation):
        # We don't actually modify the observation here, just make sure it's normalized
        # Cart position is already at index 0, and pole angle at index 2
        return observation


class DQNAgent:
    def __init__(self, state_size, action_size, global_step, log_dir, initial_model_path=None):
        self.state_size = state_size
        self.action_size = action_size
        self.global_step = global_step
        self.log_dir = log_dir
        self.initial_model_path = initial_model_path
        
        # Hyperparameters
        self.memory = deque(maxlen=20000)   # Experience replay buffer
        self.gamma = 0.9                   # Discount factor
        self.epsilon = 1.0                 # Exploration rate
        self.epsilon_min = 0.01            # Minimum exploration probability
        self.epsilon_decay = 0.995         # Exponential decay rate for exploration
        self.learning_rate = 0.001         # Learning rate
        self.batch_size = 128              # Size of batches for training
        self.train_start = 1000            # Minimum experiences before training
        self.update_target_frequency = 10  # How often to update target network (episodes)
        
        # Load or build the main model
        if initial_model_path and os.path.exists(initial_model_path):
            print(f"Loading initial model from: {initial_model_path}")
            self.model = tf.keras.models.load_model(initial_model_path)
        else:
            self.model = self._build_model()
        
        # Target model - used for more stable Q-value predictions
        self.target_model = self._build_model()
        
        # Initialize target model with same weights as main model
        self.update_target_model()
        
    
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
    
    def act(self, state, eval_mode=False):
        """Epsilon-greedy action selection, with option for pure exploitation"""
        if not eval_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values for all actions in current state
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the network using randomly sampled experiences with vectorized operations"""
        # Check if we have enough experiences
        if len(self.memory) < self.train_start:
            return 0.0  # Return 0 loss if no training performed
        
        # Sample a random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batches using vectorized operations
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Get all predictions in single batched operations
        targets = self.model.predict(states, verbose=0, batch_size=self.batch_size)
        next_q_values = self.model.predict(next_states, verbose=0, batch_size=self.batch_size)
        target_q_values = self.target_model.predict(next_states, verbose=0, batch_size=self.batch_size)
        
        # Get max actions from main model predictions (already computed)
        max_actions = np.argmax(next_q_values, axis=1)
        
        # Create action indices for efficient batch update
        batch_indices = np.arange(self.batch_size)
        
        # Vectorized update for targets
        # Start with the rewards
        target_values = rewards.copy()
        
        # For non-terminal states, add the discounted future reward
        non_terminal_mask = ~dones
        target_values[non_terminal_mask] += self.gamma * target_q_values[non_terminal_mask, max_actions[non_terminal_mask]]
        
        # Update only the specific action values using advanced indexing
        targets[batch_indices, actions] = target_values
        
        # Train the network with optimized batch processing
        history = self.model.fit(
            states, 
            targets, 
            epochs=1, 
            verbose=0, 
            batch_size=self.batch_size
        )
        
        # Increment global step and track loss
        loss = history.history['loss'][0]
            
        return loss
    
    def save_model(self, episode):
        """Save model checkpoint"""
        model_path = os.path.join(self.log_dir, f"model_episode_{episode}.keras")
        self.model.save(model_path)
        print(f"Model saved to: {model_path}")


def run_evaluation_episode(env, agent, episode_num):
    """Run a single evaluation episode with rendering and no training"""
    state, _ = env.reset()
    state = np.reshape(state, [1, agent.state_size])
    done = False
    score = 0
    rewards = []
    
    # Create rendering environment
    eval_env = gym.make('CartPole-v1', render_mode='human')
    eval_env = CartPoleCenteringWrapper(eval_env)  # Apply the wrapper
    eval_state, _ = eval_env.reset()
    eval_state = np.reshape(eval_state, [1, agent.state_size])
    
    print(f"\n--- VISUAL EVALUATION (Episode {episode_num}) ---")
    
    while not done:
        # Use model without exploration (epsilon=0)
        action = agent.act(eval_state, eval_mode=True)
        
        # Take action
        next_state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        
        # Calculate custom rewards for consistency in reporting
        MAX_ANGLE = 0.2095
        MAX_POSITION = 2.4  # Maximum cart position
        
        cart_position = next_state[0]  # Cart position
        pole_angle = next_state[2]     # Pole angle
        
        # Calculate reward components
        angle_reward = 1.0 - (abs(pole_angle) / MAX_ANGLE)
        position_reward = max(0, 1.0 - (abs(cart_position) / MAX_POSITION))  # Higher when closer to center
        
        # Combined reward (weighted 70% angle, 30% position)
        custom_reward = 0.7 * angle_reward + 0.3 * position_reward
        
        if done and score < 499:
            custom_reward = -10
            
        rewards.append(custom_reward)
        eval_state = np.reshape(next_state, [1, agent.state_size])
        score += 1
    
    avg_reward = np.mean(rewards) if rewards else 0
    print(f"Evaluation Score: {score}, Avg Reward: {avg_reward:.4f}")
    print("--- END OF EVALUATION ---\n")
    
    # Close the rendering environment
    eval_env.close()
    
    return score, avg_reward


def main():
    MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
    MAX_POSITION = 2.4  # Maximum cart position

    # Create environment with our wrapping for training
    env = gym.make('CartPole-v1')
    env = CartPoleCenteringWrapper(env)
    
    # Get environment information (state size remains the same with our wrapper)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size, global_step, log_dir, INITIAL_MODEL_PATH)
    
    # Training parameters
    EPISODES = 500
    
    # Keep track of rewards per episode
    scores = []
    avg_scores = []
    
    # Track training time
    training_start_time = time.time()
    
    for e in range(EPISODES):

        # Every 10th episode, run an evaluation with rendering, and save the model
        if (e==0 or e > 50) and e % 10 == 0:
            eval_score, eval_reward = run_evaluation_episode(env, agent, e)
            agent.save_model(e)
        
        # Reset environment for training
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        episode_rewards = []  # Track all rewards for episode statistics
        episode_loss = []     # Track all losses for episode statistics
        episode_start_time = time.time()  # Track episode start time
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Extract values for custom reward
            cart_position = next_state[0]  # Used for position component
            pole_angle = next_state[2]     # Used for angle component
            
            # Calculate custom reward components
            angle_reward = 1.0 - (abs(pole_angle) / MAX_ANGLE)  # 1.0 when vertical, 0.0 at maximum angle
            position_reward = max(0, 1.0 - (abs(cart_position) / MAX_POSITION))  # 1.0 when centered
            
            # Combined reward (weighted 70% angle, 30% position)
            custom_reward = 0.7 * angle_reward + 0.3 * position_reward
            
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
            
            # Train on past experiences (replay) after each fourth step
            if score % 4 == 0:
                episode_loss.append(agent.replay())
            
            # Increment global step
            global_step.assign_add(1)
            
            if done:
                break
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        total_training_time = time.time() - training_start_time
        
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

        # Calculate average loss per step for this episode
        avg_loss = np.mean([l for l in episode_loss if l is not None]) if episode_loss else 0

        # Update epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Format times for display
        total_hours, remainder = divmod(total_training_time, 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        episode_minutes, episode_seconds = divmod(episode_time, 60)
        
        # Print progress with training time
        print(f"Episode: {e}/{EPISODES}, Score: {score}, Avg: {avg_score:.2f}, " +
              f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.2f}, Avg Loss: {avg_loss:.4f}, Time: {int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s")
			
        # Write to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('score', score, step=e)
            tf.summary.scalar('avg_score_100', avg_score, step=e)
            tf.summary.scalar('avg_reward', avg_reward, step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)
            tf.summary.scalar('avg_loss', avg_loss, step=e)
            tf.summary.scalar('episode_time_seconds', episode_time, step=e)
        
        # If we've solved the environment (avg score >= 195 over 100 episodes), stop
        if len(scores) >= 100 and avg_score >= 195:
            print(f"Environment solved in {e} episodes! Average score: {avg_score:.2f}")
            # Save final model
            agent.save_model("final")
            break
    
    # Save final model regardless
    agent.model.save(os.path.join(log_dir, 'trained_cartpole_model_final.keras'))
    print(f"Final model saved to: {os.path.join(log_dir, 'trained_cartpole_model_final.keras')}")

    # Close environment
    env.close()
    
    # Report final training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()