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

def run_evaluation_batch(model, vec_env, state_size, n_episodes):
    """Run n_episodes in parallel using a vectorized env, return mean score and reward"""
    states, _ = vec_env.reset()
    states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)

    scores = np.zeros(n_episodes, dtype=np.int32)
    reward_sums = np.zeros(n_episodes, dtype=np.float64)
    reward_counts = np.zeros(n_episodes, dtype=np.int32)
    active = np.ones(n_episodes, dtype=bool)

    while np.any(active):
        q_values = model(states_tensor, training=False)
        actions = tf.argmax(q_values, axis=1).numpy()

        next_states, rewards, terminations, truncations, infos = vec_env.step(actions)
        dones = np.logical_or(terminations, truncations)

        # Custom reward for active envs
        cart_pos = next_states[:, 0]
        pole_angle = next_states[:, 2]
        custom_rewards = (1.0 - (np.abs(pole_angle) / MAX_ANGLE)) * 0.5 + \
                         np.maximum(0.0, 1.0 - (np.abs(cart_pos) / MAX_POSITION)) * 0.5

        scores[active] += 1
        reward_sums[active] += custom_rewards[active]
        reward_counts[active] += 1

        # Mark finished episodes as inactive
        newly_done = active & dones
        active[newly_done] = False

        states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)

    avg_rewards = np.where(reward_counts > 0, reward_sums / reward_counts, 0.0)
    return float(np.mean(scores)), float(np.mean(avg_rewards))

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
    
    # Create vectorized evaluation environment (5 parallel episodes)
    n_eval_episodes = 5
    if graphics:
        eval_env = gym.make_vec('CartPole-v1', num_envs=n_eval_episodes, render_mode='human')
    else:
        eval_env = gym.make_vec('CartPole-v1', num_envs=n_eval_episodes)

    # Main evaluation loop (runs silently - results sent over pipe)
    while True:
        try:
            message = conn.recv()
            if message == "STOP":
                break

            model_path, total_steps = message

            try:
                saved_model = tf.keras.models.load_model(model_path)
                avg_score, avg_reward = run_evaluation_batch(
                    saved_model, eval_env, state_size, n_eval_episodes
                )
                conn.send((total_steps, avg_score, avg_reward))
            except Exception as e:
                conn.send(("ERROR", f"Model loading/evaluation failed: {str(e)}"))

        except Exception as e:
            conn.send(("ERROR", str(e)))

    eval_env.close()
    conn.close()

if __name__ == "__main__":
    main()