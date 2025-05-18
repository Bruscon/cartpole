import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow import keras
from collections import deque
import random
import time
import os
import shutil
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
import argparse
import cProfile
import pstats

from DQNAgent import DQNAgent

verbose = 1

# Enable profiling
pr = cProfile.Profile()
pr.enable()

# Suppress TensorFlow warnings
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# #use mixed precision 
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

#allows GPU memory to be allocated as needed
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
MAX_POSITION = 2.4  # Maximum cart position

# Make sure we're using my beast of a GPU (NVIDIA RTX 3090 with 24GB of VRAM)
if verbose > 1:
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    print("Using GPU:", tf.config.list_physical_devices('GPU'))
    print("CUDA built:", tf.test.is_built_with_cuda())

# Parse command line arguments
parser = argparse.ArgumentParser(description='DQN training for CartPole')
parser.add_argument('--log-dir', type=str, help='Directory for logs')
parser.add_argument('--model', type=str, help='Path to initial model')
parser.add_argument('--graphics', action='store_true', default=False, help='Whether evaluation episodes should be rendered graphically')
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

class TFWrappedVecEnv:
    """Wrapper for vectorized Gym environment to reduce TF-NumPy conversions"""
    def __init__(self, vec_env):
        self.vec_env = vec_env
    
    def reset(self, seed=None):
        states, info = self.vec_env.reset(seed=seed)
        # Convert states to tensor once
        return tf.convert_to_tensor(states, dtype=tf.float32), info
    
    def step(self, actions):
        # Convert actions tensor to numpy once
        actions_np = actions.numpy() if isinstance(actions, tf.Tensor) else actions
        
        # Run environment step with numpy arrays
        next_states, rewards, terminations, truncations, infos = self.vec_env.step(actions_np)
        
        # Convert results to tensors once
        return (
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(terminations, dtype=tf.bool),
            tf.convert_to_tensor(truncations, dtype=tf.bool),
            infos
        )
    
    def close(self):
        return self.vec_env.close()

def print_memory_usage():
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"System RAM usage: {mem_info.rss / (1024 * 1024):.2f} MB")

def print_gpu_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"GPU memory allocated: {mem_info['current'] / (1024 * 1024):.2f} MB")
            print(f"GPU memory peak: {mem_info['peak'] / (1024 * 1024):.2f} MB")
        except:
            try:
                mem_info = tf.experimental.get_memory_usage('GPU:0')
                print(f"GPU memory usage: {mem_info / (1024 * 1024):.2f} MB")
            except:
                print("GPU memory usage unavailable")

def run_evaluation_episode(agent, eval_env, step_num):
    """Run a single episode with the current policy for evaluation"""
    state, _ = eval_env.reset()
    state = np.reshape(state, [1, agent.state_size])
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 500:
        # Use the policy without exploration
        action = agent.act(state, eval_mode=True)
        
        # Take action
        next_state, reward, termination, truncation, _ = eval_env.step(action)
        done = termination or truncation
        
        # Reshape and save next state
        next_state = np.reshape(next_state, [1, agent.state_size])
        
        # Update state and count
        state = next_state
        total_reward += reward
        steps += 1
    
    print(f"----EVALUATION EPISODE at step {step_num}: Score: {steps}, Reward: {total_reward}")
    return steps, total_reward

# Custom reward calculation function
@tf.function
def calculate_custom_rewards(cart_positions, pole_angles, terminations, truncations):
    """Calculate custom rewards using TensorFlow operations"""
    angle_rewards = 1.0 - (tf.abs(pole_angles) / MAX_ANGLE)
    position_rewards = tf.maximum(0.0, 1.0 - (tf.abs(cart_positions) / MAX_POSITION))
    
    # Base rewards
    custom_rewards = 0.7 * angle_rewards + 0.3 * position_rewards
    
    # Apply termination penalty
    dones = tf.logical_or(terminations, truncations)
    custom_rewards = tf.where(dones, tf.constant(-1.0, dtype=tf.float32), custom_rewards)
    
    return custom_rewards, dones


def main():
    # Create a single environment first to get state/action dimensions
    single_env = gym.make('CartPole-v1')
    state_size = single_env.observation_space.shape[0]
    action_size = single_env.action_space.n
    single_env.close()

    
    # Create environment, vectorized
    n_envs = 16  
    gym_envs = gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="async")
    # Wrap with TF adapter
    envs = TFWrappedVecEnv(gym_envs)

    # Create agent
    agent = DQNAgent(state_size, action_size, log_dir, INITIAL_MODEL_PATH)
    
    # Training hyperparameters
    TOTAL_TIMESTEPS = 10_000_000  # Total training steps
    EVAL_FREQUENCY = 100          # How often to run evaluation episodes
    SAVE_FREQUENCY = 5_000        # How often to save the model
    LOG_FREQUENCY = 25            # How often to print logs
    
    # Track metrics using TensorFlow variables where appropriate
    total_steps = 0
    episode_counts = np.zeros(n_envs, dtype=np.int32)    # Track episodes completed per env
    env_steps = np.zeros(n_envs, dtype=np.int32)         # Track steps in current episode per env
    env_rewards = np.zeros(n_envs, dtype=np.float32)     # Track rewards in current episode per env
    
    # Global statistics
    completed_episodes = 0
    
    # Track metrics between logging periods instead of using deque with fixed size
    log_period_episode_rewards = []
    log_period_episode_lengths = []
    log_period_losses = []
    
    # Track training time
    training_start_time = time.time()
    last_log_time = training_start_time

    # Main training loop - run for a fixed number of steps
    states, _ = envs.reset()

    while total_steps < TOTAL_TIMESTEPS:
        # Get actions from forward pass on NN
        greedy_actions = agent.get_greedy_actions(states)

        # Insert randomness for exploration (replaces values with a random action according to epsilon)
        actions = agent.explore_batch(greedy_actions)

        # Take actions in all environments
        next_states, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Calculate dones
        dones = tf.logical_or(terminations, truncations)
        
        # Calculate custom rewards using TensorFlow operations
        cart_positions = next_states[:, 0]
        pole_angles = next_states[:, 2]
        
        angle_rewards = 1.0 - (tf.abs(pole_angles) / MAX_ANGLE)
        position_rewards = tf.maximum(0.0, 1.0 - (tf.abs(cart_positions) / MAX_POSITION))
        
        custom_rewards = 0.7 * angle_rewards + 0.3 * position_rewards
        custom_rewards = tf.where(dones, tf.constant(-1.0, dtype=tf.float32), custom_rewards)
        
        # Store batch of experiences in replay buffer
        agent.remember_batch(states, actions, custom_rewards, next_states, dones)
        
        # Update metrics for each environment
        env_steps += 1  # Increment steps for all environments
        env_rewards += custom_rewards.numpy()  # Add rewards to running totals
        total_steps += 1

        # Handle episode terminations for each environment
        done_indices = np.where(dones.numpy())[0]
        for i in done_indices:
            completed_episodes += 1
            episode_counts[i] += 1
            
            # Store episode stats for this logging period
            log_period_episode_rewards.append(env_rewards[i])
            log_period_episode_lengths.append(env_steps[i])
            
            # Reset counters for this environment
            env_steps[i] = 0
            env_rewards[i] = 0
        
        # Update states for next iteration
        states = next_states
        
        # Train periodically 
        if total_steps % (agent.train_frequency) == 0:
            loss = agent.replay()
            # Store loss for logging
            log_period_losses.append(loss)
        
        # Update target model periodically
        if total_steps % (agent.update_target_frequency) == 0:
            agent.update_target_model()
            print("Target model updated")
        
        # Log progress periodically
        if total_steps % LOG_FREQUENCY == 0:
            current_time = time.time()
            elapsed_time = current_time - last_log_time
            total_elapsed = current_time - training_start_time
            
            # Calculate statistics from data since last log
            avg_reward = np.mean(log_period_episode_rewards) if log_period_episode_rewards else 0
            avg_length = np.mean(log_period_episode_lengths) if log_period_episode_lengths else 0
            avg_loss = np.mean(log_period_losses) if log_period_losses else 0
            steps_per_second = LOG_FREQUENCY / elapsed_time if elapsed_time > 0 else 0
            
            # Format times for display
            total_hours, remainder = divmod(total_elapsed, 3600)
            total_minutes, total_seconds = divmod(remainder, 60)
            
            print(f"Steps: {total_steps}, Episodes: {completed_episodes}, " +
                  f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, " +
                  f"Avg Loss: {avg_loss:.4f}, SPS: {steps_per_second:.1f}, " +
                  f"LR: {agent.get_current_learning_rate():.4f}, " +
                  f"ε: {agent.epsilon:.3f}, " +
                  f"Time: {int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s")
            
            # Reset the log period tracking lists after logging
            log_period_episode_rewards = []
            log_period_episode_lengths = []
            log_period_losses = []
            
            last_log_time = current_time
            
            # Occasionally print memory usage
            if total_steps % (LOG_FREQUENCY * 10) == 0:
                print_memory_usage()
                print_gpu_memory()
        
        # Run evaluation periodically
        if total_steps % EVAL_FREQUENCY == 0:
            # Create a separate evaluation environment
            eval_env = gym.make('CartPole-v1')
            eval_score, eval_reward = run_evaluation_episode(agent, eval_env, total_steps)
            eval_env.close()
            
        # Save model periodically
        if total_steps % SAVE_FREQUENCY == 0:
            agent.save_model(total_steps)
    
    # Save final model
    agent.model.save(os.path.join(log_dir, 'trained_cartpole_model_final.keras'))
    print(f"Final model saved to: {os.path.join(log_dir, 'trained_cartpole_model_final.keras')}")

    # Close environment
    tf_envs.close()
    
    # Report final training time
    total_training_time = time.time() - training_start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")


    # #profiling stats
    # stats = pstats.Stats(pr)
    # stats.sort_stats('cumtime')  # Sort by cumulative time
    # stats.print_stats(5)  # Print top 5 time-consuming functions

if __name__ == "__main__":
    main()