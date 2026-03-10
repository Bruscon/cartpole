import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gymnasium as gym
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
import time
import sys
import shutil
import json
import argparse
import subprocess
from multiprocessing.connection import Pipe

from SACAgent import SACAgent
from TrainingLogger import TrainingLogger

MAX_ANGLE = 0.2095  # Maximum angle before termination (in radians)
MAX_POSITION = 2.4  # Maximum cart position

# Allow GPU memory growth to avoid allocating all VRAM upfront
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

# Parse command line arguments
parser = argparse.ArgumentParser(description='SAC training for CartPole')
parser.add_argument('--log-dir', type=str, help='Directory for logs')
parser.add_argument('--model', type=str, help='Path to initial model')
parser.add_argument('--max-seconds', type=float, default=None,
                    help='Maximum wall-clock training time in seconds')
parser.add_argument('--graphics', action='store_true', default=False,
                    help='Whether evaluation episodes should be rendered graphically')
args = parser.parse_args()

if args.log_dir:
    log_dir = args.log_dir
else:
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_dir = f"logs/dqn_{current_time}"

INITIAL_MODEL_PATH = args.model if args.model else None

os.makedirs(log_dir, exist_ok=True)

# Archive a copy of this script to the logging folder
script_path = os.path.abspath(__file__)
shutil.copy2(script_path, os.path.join(log_dir, os.path.basename(script_path)))


class TFWrappedVecEnv:
    """Wrapper for vectorized Gym environment to reduce TF-NumPy conversions"""
    def __init__(self, vec_env):
        self.vec_env = vec_env

    def reset(self, seed=None):
        states, info = self.vec_env.reset(seed=seed)
        return tf.convert_to_tensor(states, dtype=tf.float32), info

    def step(self, actions):
        actions_np = actions.numpy() if isinstance(actions, tf.Tensor) else actions
        next_states, rewards, terminations, truncations, infos = self.vec_env.step(actions_np)
        return (
            tf.convert_to_tensor(next_states, dtype=tf.float32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(terminations, dtype=tf.bool),
            tf.convert_to_tensor(truncations, dtype=tf.bool),
            infos
        )

    def close(self):
        return self.vec_env.close()


def get_memory_stats():
    """Return RAM and GPU memory stats as a dict (MB)"""
    import psutil
    stats = {}
    process = psutil.Process(os.getpid())
    stats['ram_mb'] = round(process.memory_info().rss / (1024 * 1024), 1)
    if gpus:
        try:
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            stats['gpu_mb'] = round(mem_info['current'] / (1024 * 1024), 1)
        except Exception:
            pass
    return stats


@tf.function
def calculate_custom_rewards(cart_positions, pole_angles, terminations, truncations):
    """Calculate custom rewards using TensorFlow operations"""
    angle_rewards = 1.0 - (tf.abs(pole_angles) / MAX_ANGLE)
    position_rewards = tf.maximum(0.0, 1.0 - (tf.abs(cart_positions) / MAX_POSITION))
    custom_rewards = angle_rewards * position_rewards
    dones = tf.logical_or(terminations, truncations)
    return custom_rewards, dones


def emit(record: dict):
    """Print a JSON line to stdout and flush"""
    print(json.dumps(record), flush=True)


def main():
    single_env = gym.make('CartPole-v1')
    state_size = single_env.observation_space.shape[0]
    action_size = single_env.action_space.n
    single_env.close()

    n_envs = 64
    gym_envs = gym.make_vec("CartPole-v1", num_envs=n_envs, vectorization_mode="async")
    envs = TFWrappedVecEnv(gym_envs)

    agent = SACAgent(state_size, action_size, log_dir, INITIAL_MODEL_PATH)
    logger = TrainingLogger(log_dir, window_size=25)

    TOTAL_TIMESTEPS = 10_000_000
    EVAL_FREQUENCY = 10
    SAVE_FREQUENCY = 5_000
    LOG_FREQUENCY = 25

    total_steps = 0
    episode_counts = np.zeros(n_envs, dtype=np.int32)
    env_steps = np.zeros(n_envs, dtype=np.int32)
    env_rewards = np.zeros(n_envs, dtype=np.float32)
    completed_episodes = 0

    training_start_time = time.time()
    evaluation_pending = False
    last_eval_step = 0
    eval_scores = []  # Track eval scores for rolling convergence check
    convergence_step = None  # Step at which convergence criterion was met

    parent_conn, child_conn = Pipe()
    os.makedirs(os.path.join(log_dir, "eval_models"), exist_ok=True)

    eval_worker_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_worker.py")
    graphics_str = "true" if args.graphics else "false"

    eval_process = subprocess.Popen([
        sys.executable,
        eval_worker_path,
        str(state_size),
        str(action_size),
        log_dir,
        graphics_str,
        str(child_conn.fileno()),
    ], pass_fds=[child_conn.fileno()])
    child_conn.close()

    emit({"type": "start", "log_dir": log_dir, "n_envs": n_envs,
          "total_timesteps": TOTAL_TIMESTEPS, "agent": "SAC"})

    states, _ = envs.reset(seed=42)
    deadline = time.monotonic() + args.max_seconds if args.max_seconds is not None else None

    try:
        while total_steps < TOTAL_TIMESTEPS:
            if deadline is not None and time.monotonic() >= deadline:
                break

            actions = agent.get_actions(states)

            next_states, rewards, terminations, truncations, infos = envs.step(actions)
            custom_rewards, dones = calculate_custom_rewards(
                next_states[:, 0], next_states[:, 2], terminations, truncations
            )

            agent.memory.add(states, actions, custom_rewards, next_states, dones)

            env_steps += 1
            env_rewards += custom_rewards.numpy()
            total_steps += 1

            logger.log_metrics(
                step=total_steps,
                lr=agent.get_current_learning_rate(),
                epsilon=agent.epsilon,
                alpha=agent.alpha,
                beta=agent.memory.beta,
            )

            done_indices = np.where(dones.numpy())[0]
            for i in done_indices:
                completed_episodes += 1
                episode_counts[i] += 1
                logger.log_metrics(
                    step=total_steps,
                    reward=env_rewards[i],
                    length=env_steps[i]
                )
                env_steps[i] = 0
                env_rewards[i] = 0

            states = next_states

            if total_steps % agent.train_frequency == 0:
                loss, avg_td_error = agent.replay()
                logger.log_metrics(step=total_steps, loss=loss, avg_td_error=avg_td_error)
                agent.update_target_models(tau=agent.tau)

            # Structured progress log
            if total_steps % LOG_FREQUENCY == 0:
                elapsed = time.time() - training_start_time
                mem_stats = get_memory_stats() if total_steps % (LOG_FREQUENCY * 10) == 0 else {}
                emit({
                    "type": "progress",
                    "step": int(total_steps),
                    "elapsed_s": round(elapsed, 1),
                    "episodes": int(completed_episodes),
                    "alpha": round(float(agent.alpha), 4),
                    "lr": round(float(agent.get_current_learning_rate()), 6),
                    "mem_size": int(agent.memory.current_size),
                    **mem_stats,
                })

            # Trigger evaluation
            if total_steps % EVAL_FREQUENCY == 0 and not evaluation_pending:
                agent.save_model(completed_episodes)
                model_path = os.path.join(
                    agent.log_dir, f"model_episode_{completed_episodes}_policy.keras"
                )
                parent_conn.send((model_path, total_steps))
                evaluation_pending = True
                last_eval_step = total_steps

            # Collect evaluation results
            if evaluation_pending and parent_conn.poll():
                try:
                    eval_result = parent_conn.recv()
                    if eval_result[0] == "ERROR":
                        emit({"type": "eval_error", "msg": eval_result[1]})
                    else:
                        eval_step, eval_score, eval_reward = eval_result
                        logger.log_metrics(step=eval_step, eval_reward=eval_reward * eval_score)
                        eval_scores.append(eval_score)
                        # Check rolling convergence: last 10 eval checkpoints avg >= 475
                        rolling_avg = np.mean(eval_scores[-10:]) if len(eval_scores) >= 10 else 0.0
                        record = {
                            "type": "eval",
                            "step": int(eval_step),
                            "score": round(float(eval_score), 1),
                            "reward": round(float(eval_reward * eval_score), 4),
                            "rolling_avg": round(float(rolling_avg), 1),
                        }
                        if convergence_step is None and len(eval_scores) >= 10 and rolling_avg >= 475:
                            convergence_step = int(eval_step)
                            record["converged"] = True
                            record["convergence_step"] = convergence_step
                        emit(record)
                        logger.log_json(record)
                    evaluation_pending = False
                except Exception as e:
                    emit({"type": "eval_error", "msg": str(e)})

            # Hard target update
            if total_steps % agent.update_target_frequency == 0:
                agent.update_target_models(tau=1.0)
                emit({"type": "target_hard_update", "step": int(total_steps)})

            # Save plot periodically
            logger.maybe_update_plot(
                force=(total_steps % (LOG_FREQUENCY * 5) == 0),
                save=(total_steps % SAVE_FREQUENCY == 0)
            )

        agent.save_model(completed_episodes)
        done_record = {"type": "done", "step": int(total_steps), "episodes": int(completed_episodes)}
        if convergence_step is not None:
            done_record["convergence_step"] = convergence_step
        emit(done_record)
        logger.update_plot(save=True)
        logger.save_data()

    except KeyboardInterrupt:
        emit({"type": "interrupted", "step": int(total_steps)})
    except Exception as e:
        emit({"type": "error", "msg": str(e)})
        raise
    finally:
        try:
            parent_conn.send("STOP")
            time.sleep(0.5)
        except Exception:
            pass

        if eval_process.poll() is None:
            eval_process.terminate()

        parent_conn.close()
        envs.close()
        logger.close()

        elapsed = time.time() - training_start_time
        emit({"type": "shutdown", "elapsed_s": round(elapsed, 1)})


if __name__ == "__main__":
    main()
