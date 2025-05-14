import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import TensorBoard
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

from TFReplayBuffer import TFReplayBuffer
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

# Create TensorBoard writer
summary_writer = tf.summary.create_file_writer(log_dir)


def run_evaluation_episode(env, agent, episode_num):
    """Run a single evaluation episode with rendering and no training"""
    state, _ = env.reset()
    state = np.reshape(state, [1, agent.state_size])
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
        
        # if done and score < 499:
        #     custom_reward = -1
            
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

    # Create environment with our wrapping for training
    env = gym.make('CartPole-v1')
    
    # Get environment information (state size remains the same with our wrapper)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size, log_dir, INITIAL_MODEL_PATH)
    
    EPISODES = 100000
    
    # Keep track of rewards per episode
    scores = []
    avg_scores = []
    
    # Track training time
    training_start_time = time.time()
    
    for e in range(EPISODES):

        # Every 10th episode, run an evaluation and save the model
        if e % 10 == 0:
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
                custom_reward = -1
            
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience with custom reward
            agent.remember(state, action, custom_reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Track rewards
            episode_rewards.append(custom_reward)
            score += 1
            
            # Train on past experiences (replay) every X steps
            if score % 8 == 0:
                episode_loss.append(agent.replay())
            
            if done:
                break
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        total_training_time = time.time() - training_start_time
        
        # Update target model periodically
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()
            print("Target model updated")
        
        # Save scores
        scores.append(score)
        
        # Calculate average score of last 25 episodes
        avg_score = np.mean(scores[-25:])
        avg_scores.append(avg_score)
        
        # Calculate average reward per step for this episode
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0 

        # Calculate average loss per step for this episode
        avg_loss = np.mean([l for l in episode_loss if l is not None]) if episode_loss else 0
        
        # Format times for display
        total_hours, remainder = divmod(total_training_time, 3600)
        total_minutes, total_seconds = divmod(remainder, 60)
        episode_minutes, episode_seconds = divmod(episode_time, 60)
        
        # Print progress with training time
        print(f"Episode: {e}, Score: {score}, Avg: {avg_score:.2f}, " +
               f"Learn Rate: {agent.get_current_learning_rate():.4f}, Epsilon: {agent.epsilon:.3f}, Avg Loss: {avg_loss:.4f}, " +
               f"Time: {int(total_hours)}h {int(total_minutes)}m {total_seconds:.2f}s")
        #f"Avg Reward: {avg_reward:.2f},
			
        # Write to TensorBoard
        with summary_writer.as_default():
            tf.summary.scalar('score', score, step=e)
            tf.summary.scalar('avg_score_100', avg_score, step=e)
            tf.summary.scalar('avg_reward', avg_reward, step=e)
            tf.summary.scalar('epsilon', agent.epsilon, step=e)
            tf.summary.scalar('avg_loss', avg_loss, step=e)
            tf.summary.scalar('episode_time_seconds', episode_time, step=e)

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