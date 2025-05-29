# evaluation_worker.py - Standalone version without DQNAgent dependency

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os
import sys
import time

# Constants needed for reward calculation
MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
MAX_POSITION = 2.4  # Maximum cart position

def run_evaluation_episode(model, eval_env, state_size):
    """Run a single evaluation episode with the provided model and environment"""
    done = False
    score = 0
    rewards = []

    # Reset the evaluation environment
    eval_state, _ = eval_env.reset()
    eval_state = np.reshape(eval_state, [1, state_size])
    eval_state_tensor = tf.convert_to_tensor(eval_state, dtype=tf.float32)

    while not done:
        # Get action from model (greedy)
        q_values = model(eval_state_tensor, training=False)
        action = tf.argmax(q_values, axis=1).numpy()[0]
        
        # Take action
        next_state, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        # Calculate custom reward
        cart_pos = np.array([next_state[0]])
        pole_angle = np.array([next_state[2]])
        custom_reward = (1.0 - (np.abs(pole_angle) / MAX_ANGLE)) * 0.5 + \
                        (np.maximum(0.0, 1.0 - (np.abs(cart_pos) / MAX_POSITION))) * 0.5

        rewards.append(custom_reward[0])
        eval_state = np.reshape(next_state, [1, state_size])
        eval_state_tensor = tf.convert_to_tensor(eval_state, dtype=tf.float32)
        score += 1

    avg_reward = np.mean(rewards) if rewards else 0
    return score, avg_reward

# This function is run directly when this script is executed independently
def main():
    if len(sys.argv) < 5:
        print("Usage: python evaluation_worker.py <state_size> <action_size> <log_dir> <graphics> <mp_conn_fd>")
        sys.exit(1)
    
    # Parse arguments
    state_size = int(sys.argv[1])
    action_size = int(sys.argv[2])
    log_dir = sys.argv[3]
    graphics = sys.argv[4].lower() == 'true'
    mp_conn_fd = int(sys.argv[5])  # File descriptor for the connection
    
    # Set up connection to parent process
    import multiprocessing.connection as connection
    conn = connection.Connection(mp_conn_fd)
    
    # Configure the process to avoid using all GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Limit TensorFlow to only allocate a portion of GPU memory
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]  # Allocate only 1GB
            )
        except RuntimeError as e:
            print(f"GPU config error: {e}")
    
    # Create evaluation environment
    if graphics:
        eval_env = gym.make('CartPole-v1', render_mode='human')
    else:
        eval_env = gym.make('CartPole-v1')
    
    print(f"Evaluation worker started, waiting for evaluation requests...")
    
    # Main evaluation loop
    while True:
        try:
            # Get a model path from the connection (blocking call)
            message = conn.recv()
            
            # Check if we received a termination signal
            if message == "STOP":
                print("Evaluation worker received stop signal")
                break
            
            # Unpack evaluation data
            model_path, total_steps = message
            
            print(f"Received evaluation request for model at step {total_steps}")
            
            # Load the model directly
            try:
                saved_model = tf.keras.models.load_model(model_path)
                
                # Run the evaluation episode
                eval_score, eval_reward = run_evaluation_episode(saved_model, eval_env, state_size)
                
                # Send results back to main process
                conn.send((total_steps, eval_score, eval_reward))
                
                print(f"Evaluation completed at step {total_steps}: Score {eval_score}, Reward {eval_reward:.4f}")
                
            except Exception as e:
                print(f"Error loading or evaluating model {model_path}: {e}")
                conn.send(("ERROR", f"Model loading/evaluation failed: {str(e)}"))
            
        except Exception as e:
            print(f"Error in evaluation worker: {e}")
            # Put the error in the result queue to inform the main process
            conn.send(("ERROR", str(e)))
    
    # Clean up
    eval_env.close()
    conn.close()
    print("Evaluation worker stopped")

if __name__ == "__main__":
    main()