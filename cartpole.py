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
import os, sys
import shutil
from gymnasium import spaces
from gymnasium.core import ObservationWrapper
import argparse
import cProfile
import pstats
import subprocess
from multiprocessing.connection import Pipe
from pathlib import Path

from DQNAgent import DQNAgent
from TrainingLogger import TrainingLogger
# from evaluation_worker import evaluation_worker_main

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

# Function to calculate custom rewards
@tf.function
def calculate_custom_rewards(cart_positions, pole_angles, terminations, truncations):
    """Calculate custom rewards using TensorFlow operations"""
    angle_rewards = 1.0 - (tf.abs(pole_angles) / MAX_ANGLE)
    position_rewards = tf.maximum(0.0, 1.0 - (tf.abs(cart_positions) / MAX_POSITION))
    
    # Base rewards
    custom_rewards = 0.5 * angle_rewards + 0.5 * position_rewards
    
    # Apply termination penalty
    dones = tf.logical_or(terminations, truncations)
    # custom_rewards = tf.where(dones, tf.constant(-1.0, dtype=tf.float32), custom_rewards)
    
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

    #Logging object for generating charts
    logger = TrainingLogger(log_dir, window_size=25)
    
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
    all_episode_rewards = []
    all_episode_lengths = []
    all_losses = []

    # Track training time
    training_start_time = time.time()
    last_log_time = training_start_time

    # Stuff for evaluations
    evaluation_pending = False
    last_eval_step = 0

    # Create a pipe for communication
    parent_conn, child_conn = Pipe()
    
    # Create evaluation models directory
    eval_models_dir = os.path.join(log_dir, "eval_models")
    os.makedirs(eval_models_dir, exist_ok=True)
    
    # Start the evaluation worker as a separate process using subprocess
    eval_worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_worker.py")
    
    # Convert graphics flag to string for command line
    graphics_str = "true" if args.graphics else "false"
    
    # Launch the worker as a separate Python process
    eval_process = subprocess.Popen([
        sys.executable,  # Python interpreter
        eval_worker_path,
        str(state_size),
        str(action_size),
        log_dir,
        graphics_str,
        str(child_conn.fileno())  # Pass the file descriptor
    ], 
    pass_fds=[child_conn.fileno()])  # Pass the file descriptor to the child process
    
    # Close the child end of the pipe in the parent process
    child_conn.close()

    # Main training loop - run for a fixed number of steps
    states, _ = envs.reset()

    try:
        while total_steps < TOTAL_TIMESTEPS:
            # Get actions from forward pass on NN
            greedy_actions = agent.get_greedy_actions(states)

            # Insert randomness for exploration
            actions = agent.explore_batch(greedy_actions)

            # Take actions in all environments
            next_states, rewards, terminations, truncations, infos = envs.step(actions)

            custom_rewards, dones = calculate_custom_rewards(next_states[:,0], next_states[:,2], terminations, truncations)
            
            # Store batch of experiences in replay buffer
            agent.remember_batch(states, actions, custom_rewards, next_states, dones)
            
            # Update metrics for each environment
            env_steps += 1  # Increment steps for all environments
            env_rewards += custom_rewards.numpy()  # Add rewards to running totals
            total_steps += 1

            # Log metrics
            logger.log_metrics(
                step=total_steps,
                lr=agent.get_current_learning_rate(),
                epsilon=agent.epsilon
            )

            # Handle episode terminations for each environment
            done_indices = np.where(dones.numpy())[0]
            for i in done_indices:
                completed_episodes += 1
                episode_counts[i] += 1
                
                # Log episode metrics
                logger.log_metrics(
                    step=total_steps,
                    reward=env_rewards[i],
                    length=env_steps[i]
                )
                    
                # Reset counters for this environment
                env_steps[i] = 0
                env_rewards[i] = 0
            
            # Update states for next iteration
            states = next_states
            
            # Train periodically 
            if total_steps % agent.train_frequency == 0:
                loss = agent.replay()
                logger.log_metrics(step=total_steps, loss=loss)

                #soft update the target model
                agent.update_target_model(tau=agent.tau)
            
            
            # Occasionally print memory usage
            if total_steps % (LOG_FREQUENCY) == 0:
                total_training_time = time.time() - training_start_time
                hours, remainder = divmod(total_training_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
                print("Steps: " + str(total_steps))
                print("Memory: "+ str(int(agent.memory.current_size)) + " / " + str(int(agent.memory.buffer_size)))

            if total_steps % (LOG_FREQUENCY * 10) == 0:
                print_memory_usage()
                print_gpu_memory()
            
            # Check if we need to trigger a new evaluation
            if total_steps % EVAL_FREQUENCY == 0 and not evaluation_pending:
                agent.save_model(completed_episodes)
                
                # Request evaluation through the pipe
                model_path = os.path.join(agent.log_dir, f"model_episode_{completed_episodes}.keras")
                parent_conn.send((model_path, total_steps))
                evaluation_pending = True
                last_eval_step = total_steps
            
            # Check if evaluation results are available
            if evaluation_pending and parent_conn.poll():
                try:
                    eval_result = parent_conn.recv()
                    
                    # Check if this is an error
                    if eval_result[0] == "ERROR":
                        print(f"Error in evaluation process: {eval_result[1]}")
                    else:
                        # Unpack results
                        eval_step, eval_score, eval_reward = eval_result
                        
                        # Log evaluation results
                        logger.log_metrics(step=eval_step, eval_length=eval_score)
                        print(f"--- EVALUATION at step {eval_step}: Score: {eval_score}, Avg Reward: {eval_reward:.4f}")
                    
                    # Mark evaluation as completed
                    evaluation_pending = False
                except Exception as e:
                    print(f"Error processing evaluation result: {e}")

            # HARD update target model periodically
            if total_steps % agent.update_target_frequency == 0:
                agent.update_target_model(tau=1.0)
                print("Target model hard updated")
            
            # Update plot periodically based on time rather than steps
            logger.maybe_update_plot(
                force=(total_steps % (LOG_FREQUENCY * 5) == 0),
                save=(total_steps % SAVE_FREQUENCY == 0)
            )
        
        # Save final model
        final_model_path = os.path.join(log_dir, 'trained_cartpole_model_final.keras')
        agent.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        # Final plot update and save
        logger.update_plot(save=True)
        logger.save_data()
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Update cleanup code
        print("Stopping evaluation process...")
        # Send stop signal through the pipe
        try:
            parent_conn.send("STOP")
            # Give it some time to clean up
            time.sleep(1)
        except:
            pass
        
        # Check if process is still running and terminate if needed
        if eval_process.poll() is None:  # None means still running
            print("Terminating evaluation process...")
            eval_process.terminate()
        
        # Close the connection
        parent_conn.close()
        
        # Close environment
        envs.close()
        
        # Report final training time
        total_training_time = time.time() - training_start_time
        hours, remainder = divmod(total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

if __name__ == "__main__":
    main()