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

def run_evaluation_episode(agent, episode_num):
    """Run a single evaluation episode with rendering and no training"""
    done = False
    score = 0
    rewards = []
    
    # Create rendering environment if graphics are enabled
    if args.graphics:
        eval_env = gym.make('CartPole-v1', render_mode='human')
    else:
        eval_env = gym.make('CartPole-v1')

    eval_state, _ = eval_env.reset()
    eval_state = np.reshape(eval_state, [1, agent.state_size])
    
    while not done:
        # Use model without exploration (epsilon=0)
        actions_tensor = agent.act_batch(tf.convert_to_tensor(eval_state, dtype=tf.float32), eval_mode=True)
        action = actions_tensor.numpy()
        
        # Take action
        next_state, reward, terminated, truncated, _ = eval_env.step(action[0])
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
            custom_reward = -1
            
        rewards.append(custom_reward)
        eval_state = np.reshape(next_state, [1, agent.state_size])
        score += 1
    
    avg_reward = np.mean(rewards) if rewards else 0
    print(f"--- EVALUATION Score: {score}, Avg Reward: {avg_reward:.4f}")
    
    eval_env.close()
    
    return score, avg_reward


def main():
    MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
    MAX_POSITION = 2.4  # Maximum cart position

    # Create a single environment first to get state/action dimensions
    single_env = gym.make('CartPole-v1')
    state_size = single_env.observation_space.shape[0]
    action_size = single_env.action_space.n
    single_env.close()
    
    # Create environment, vectorized
    n_envs = 16  
    envs = gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="async")
    states, _ = envs.reset(seed=1)
    
    # Create agent
    agent = DQNAgent(state_size, action_size, log_dir, INITIAL_MODEL_PATH)
    
    EPISODES = 100000
    
    # Keep track of average rewards per episode
    avg_score = 0       # average from all environments on this one episode
    avg_scores = []     # running list from each episode
    running_avg_score=0 # average of last 25 episode averages
    
    # Track training time
    training_start_time = time.time()
    
    for e in range(EPISODES):

        # Every 10th episode, run an evaluation and save the model
        if e % 10 == 0:
            eval_score, eval_reward = run_evaluation_episode( agent, e)
            agent.save_model(e)

        # Reset environment for training
        states, _ = envs.reset()
        dones = np.zeros(n_envs, dtype=bool)
        scores = np.zeros(n_envs)
        episode_rewards = []  # Track all rewards for episode statistics
        episode_loss = []     # Track all losses for episode statistics
        episode_start_time = time.time()  # Track episode start time

        # # Track steps per environment
        # steps_per_env = np.zeros(n_envs, dtype=int)
        total_steps = 0
        # max_steps = 500  # Maximum steps per episode
        
        # # An episode is now basically just 500 timesteps in each env that resets when the bar falls
        # target_steps_per_episode = max_steps*n_envs

        while True:

            # Get actions for all environments
            actions_tensor = agent.act_batch(tf.convert_to_tensor(states, dtype=tf.float32), eval_mode=False)
            actions = actions_tensor.numpy()

            # Take actions in all environments
            next_states, rewards, terminations, truncations, infos = envs.step(actions)
            dones = np.logical_or(terminations, truncations)
            
            # Calculate custom rewards (vectorized)
            cart_positions = next_states[:, 0]
            pole_angles = next_states[:, 2]
            
            angle_rewards = 1.0 - (np.abs(pole_angles) / MAX_ANGLE)
            position_rewards = np.maximum(0, 1.0 - (np.abs(cart_positions) / MAX_POSITION))
            
            custom_rewards = 0.7 * angle_rewards + 0.3 * position_rewards

            # Apply termination penalty - note that environments may auto-reset
            # so we need to check the infos for terminal_observation if available
            if "terminal_observation" in infos:
                for i, term_obs in enumerate(infos["terminal_observation"]):
                    if terminations[i] or truncations[i]:
                        # Only apply penalty if it wasn't max steps
                        # Check episode length from infos if available
                        custom_rewards[i] = -1.0
            
            # Store batch of experiences
            agent.remember_batch(states, actions, custom_rewards, next_states, dones)
            
            # Update states
            states = next_states
            
            # Update scores and track rewards
            scores += 1  # Increment scores for each environment
            episode_rewards.extend(custom_rewards.tolist())
            total_steps += 1
            
            # Train periodically
            if total_steps % 8 == 0:
                loss = agent.replay()
                episode_loss.append(loss)
            
        # Calculate episode metrics (use mean across environments)
        avg_score = np.mean(scores)
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        total_training_time = time.time() - training_start_time
        
        # Update target model periodically
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()
            print("Target model updated")
        
        # Save scores
        avg_scores.append(avg_score)
        
        # Calculate average score of last 25 episodes
        running_avg_score = np.mean(avg_scores[-25:])
        
        # Calculate average reward per step for this episode
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0 

        # Calculate total reward for this episode
        tot_reward = np.sum(episode_rewards) if episode_rewards else 0

        # Calculate average loss per step for this episode
        avg_loss = np.mean([l for l in episode_loss if l is not None]) if episode_loss else 0
        
        # Format times for display
        total_hours, remainder = divmod(total_training_time, 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        episode_minutes, episode_seconds = divmod(episode_time, 60)
        
        # Print progress with training time
        print(f"Episode: {e}, Score: {avg_score}, Avg Reward: {tot_reward:.0f}" +
               f"Learn Rate: {agent.get_current_learning_rate():.4f}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}, " +
               f"Time: {int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s")
        #f"Avg Reward: {avg_reward:.2f},
        #Avg: {running_avg_score:.2f}, 
			
        if e % 10 == 0:
            print_memory_usage()
            print_gpu_memory()

        # #profiling stats
        # stats = pstats.Stats(pr)
        # stats.sort_stats('cumtime')  # Sort by cumulative time
        # stats.print_stats(5)  # Print top 5 time-consuming functions
    
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