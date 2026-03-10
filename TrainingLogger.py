import matplotlib
matplotlib.use('Agg')  # Headless backend - no display required
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PathCollection
import numpy as np
import os
import json
import pandas as pd
import time
from collections import deque

class TrainingLogger:
    def __init__(self, log_dir, window_size=25):
        self.log_dir = log_dir
        self.window_size = window_size

        # Create DataFrame to store all metrics
        self.data = pd.DataFrame(columns=[
            'episode_reward',
            'episode_length',
            'eval_reward',
            'loss',
            'learning_rate',
            'epsilon',
            'alpha',
            'beta',
            'avg_td_error',
            'avg_reward',
            'avg_length'
        ])

        # Create rolling windows for episode metrics
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)

        # JSON metrics file for structured output
        self.metrics_file = open(os.path.join(log_dir, 'metrics.jsonl'), 'a')

        # Setup for non-blocking updates
        self.last_update_time = time.time()
        self.update_interval = 30.0  # Save plot every 30 seconds

        # Tracking for efficiency
        self.is_first_plot = True
        self.last_plotted_indices = {
            'episode_reward': -1,
            'episode_length': -1,
            'eval_reward': -1,
            'avg_reward': -1,
            'avg_length': -1,
            'loss': -1,
            'learning_rate': -1,
            'epsilon': -1,
            'alpha': -1,
            'beta': -1,
            'avg_td_error': -1
        }

        # Create plot figure and axes
        self.setup_plot()

    def log_json(self, record: dict):
        """Write a JSON line to metrics file and flush"""
        self.metrics_file.write(json.dumps(record) + '\n')
        self.metrics_file.flush()

    def setup_plot(self):
        """Setup matplotlib figure and axes with all plot objects initialized once"""
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # First subplot - Episode metrics
        self.episode_ax = self.axes[0]
        self.episode_ax.set_title('Episode Metrics')
        self.episode_ax.set_ylabel('Episode Length / Reward')
        self.episode_ax.set_ylim(0, 500)

        # Second subplot - Training metrics
        self.training_ax = self.axes[1]
        self.training_ax.set_title('Training Metrics')
        self.training_ax.set_xlabel('Step')
        self.training_ax.set_ylabel('Loss / Epsilon')
        self.training_ax.set_ylim(-1, 2)

        # Create a second y-axis for learning rate
        self.lr_ax = self.training_ax.twinx()
        self.lr_ax.set_ylabel('Learning Rate')
        self.lr_ax.set_ylim(0, 0.1)

        # Initialize all plot objects ONCE (empty at first)
        self.reward_scatter = self.episode_ax.scatter([], [],
                                color='blue', alpha=0.5, s=10)
        self.length_scatter = self.episode_ax.scatter([], [],
                                color='green', alpha=0.5, s=10)

        self.eval_line, = self.episode_ax.plot([], [], 'r-', label='Evaluation Reward')
        self.avg_reward_line, = self.episode_ax.plot([], [], 'b-', linewidth=2,
                                    label=f'Avg Reward ({self.window_size} ep)')
        self.avg_length_line, = self.episode_ax.plot([], [], 'g-', linewidth=2,
                                    label=f'Avg Length ({self.window_size} ep)')

        self.loss_line, = self.training_ax.plot([], [], 'r-', label='Loss')
        self.epsilon_line, = self.training_ax.plot([], [], 'g-', label='Epsilon')
        self.lr_line, = self.lr_ax.plot([], [], 'b-', label='Learning Rate')
        self.alpha_line, = self.training_ax.plot([], [], 'm-', label='Alpha')
        self.beta_line, = self.training_ax.plot([], [], 'c-', label='Beta')
        self.td_error_line, = self.training_ax.plot([], [], 'orange', label='Avg TD Error')

        self.ep_artists = [self.reward_scatter, self.length_scatter,
                        self.eval_line, self.avg_reward_line, self.avg_length_line]
        self.ep_labels = ['Episode Reward', 'Episode Length', 'Eval Reward',
                        f'Avg Reward ({self.window_size} ep)',
                        f'Avg Length ({self.window_size} ep)']
        self.episode_ax.legend(self.ep_artists, self.ep_labels, loc='upper left')

        all_artists = [self.loss_line, self.epsilon_line, self.lr_line,
                       self.alpha_line, self.beta_line, self.td_error_line]
        all_labels = ['Loss', 'Epsilon', 'Learning Rate', 'Alpha', 'Beta', 'Avg TD Error']
        self.training_ax.legend(all_artists, all_labels, loc='upper left')

        plt.tight_layout()

    def log_metrics(self, step, reward=None, length=None, eval_reward=None, loss=None,
                    lr=None, epsilon=None, alpha=None, beta=None, avg_td_error=None):
        """Log all metrics at once for a given step"""
        if reward is not None:
            self.reward_window.append(reward)
        if length is not None:
            self.length_window.append(length)

        avg_reward = np.mean(self.reward_window) if self.reward_window else np.nan
        avg_length = np.mean(self.length_window) if self.length_window else np.nan

        new_data = {
            'episode_reward': reward,
            'episode_length': length,
            'eval_reward': eval_reward,
            'loss': loss,
            'learning_rate': lr,
            'epsilon': epsilon,
            'alpha': alpha,
            'beta': beta,
            'avg_td_error': avg_td_error,
            'avg_reward': avg_reward if not np.isnan(avg_reward) else None,
            'avg_length': avg_length if not np.isnan(avg_length) else None
        }

        new_data = {k: v for k, v in new_data.items() if v is not None}

        if step in self.data.index:
            for column, value in new_data.items():
                self.data.at[step, column] = value
        else:
            new_row = pd.DataFrame([new_data], index=[step])
            self.data = pd.concat([self.data, new_row])

        self.data = self.data.sort_index()

    def update_plot(self, save=False):
        """Update the plot with current data"""
        if self.data.empty:
            return

        changed = False

        reward_data = self.data['episode_reward'].dropna()
        if not reward_data.empty and reward_data.index.max() > self.last_plotted_indices['episode_reward']:
            self.reward_scatter.set_offsets(np.column_stack([reward_data.index.values, reward_data.values]))
            self.last_plotted_indices['episode_reward'] = reward_data.index.max()
            changed = True

        length_data = self.data['episode_length'].dropna()
        if not length_data.empty and length_data.index.max() > self.last_plotted_indices['episode_length']:
            self.length_scatter.set_offsets(np.column_stack([length_data.index.values, length_data.values]))
            self.last_plotted_indices['episode_length'] = length_data.index.max()
            changed = True

        eval_reward_data = self.data['eval_reward'].dropna()
        if not eval_reward_data.empty and eval_reward_data.index.max() > self.last_plotted_indices['eval_reward']:
            self.eval_line.set_data(eval_reward_data.index, eval_reward_data.values)
            self.last_plotted_indices['eval_reward'] = eval_reward_data.index.max()
            changed = True

        avg_reward_data = self.data['avg_reward'].dropna()
        if not avg_reward_data.empty and avg_reward_data.index.max() > self.last_plotted_indices['avg_reward']:
            self.avg_reward_line.set_data(avg_reward_data.index, avg_reward_data.values)
            self.last_plotted_indices['avg_reward'] = avg_reward_data.index.max()
            changed = True

        avg_length_data = self.data['avg_length'].dropna()
        if not avg_length_data.empty and avg_length_data.index.max() > self.last_plotted_indices['avg_length']:
            self.avg_length_line.set_data(avg_length_data.index, avg_length_data.values)
            self.last_plotted_indices['avg_length'] = avg_length_data.index.max()
            changed = True

        loss_data = self.data['loss'].dropna()
        if not loss_data.empty and loss_data.index.max() > self.last_plotted_indices['loss']:
            self.loss_line.set_data(loss_data.index, loss_data.values)
            self.last_plotted_indices['loss'] = loss_data.index.max()
            changed = True

        epsilon_data = self.data['epsilon'].dropna()
        if not epsilon_data.empty and epsilon_data.index.max() > self.last_plotted_indices['epsilon']:
            self.epsilon_line.set_data(epsilon_data.index, epsilon_data.values)
            self.last_plotted_indices['epsilon'] = epsilon_data.index.max()
            changed = True

        lr_data = self.data['learning_rate'].dropna()
        if not lr_data.empty and lr_data.index.max() > self.last_plotted_indices['learning_rate']:
            self.lr_line.set_data(lr_data.index, lr_data.values)
            self.last_plotted_indices['learning_rate'] = lr_data.index.max()
            changed = True

        alpha_data = self.data['alpha'].dropna()
        if not alpha_data.empty and alpha_data.index.max() > self.last_plotted_indices['alpha']:
            self.alpha_line.set_data(alpha_data.index, alpha_data.values)
            self.last_plotted_indices['alpha'] = alpha_data.index.max()
            changed = True

        beta_data = self.data['beta'].dropna()
        if not beta_data.empty and beta_data.index.max() > self.last_plotted_indices['beta']:
            self.beta_line.set_data(beta_data.index, beta_data.values)
            self.last_plotted_indices['beta'] = beta_data.index.max()
            changed = True

        td_error_data = self.data['avg_td_error'].dropna()
        if not td_error_data.empty and td_error_data.index.max() > self.last_plotted_indices['avg_td_error']:
            self.td_error_line.set_data(td_error_data.index, td_error_data.values)
            self.last_plotted_indices['avg_td_error'] = td_error_data.index.max()
            changed = True

        if changed or self.is_first_plot:
            self.episode_ax.relim()
            self.episode_ax.autoscale_view()
            self.training_ax.relim()
            self.training_ax.autoscale_view()
            self.lr_ax.relim()
            self.lr_ax.autoscale_view()
            self.fig.canvas.draw_idle()
            self.is_first_plot = False

        if save:
            plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'))

    def maybe_update_plot(self, force=False, save=False):
        """Update plot if enough time has passed or if forced"""
        current_time = time.time()
        if force or (current_time - self.last_update_time) >= self.update_interval:
            self.update_plot(save=save)
            self.last_update_time = current_time
            return True
        return False

    def save_data(self):
        """Save metrics data to CSV file"""
        if not self.data.empty:
            self.data.to_csv(os.path.join(self.log_dir, 'training_metrics.csv'))

    def close(self):
        """Flush and close the metrics file"""
        self.metrics_file.flush()
        self.metrics_file.close()
        plt.close(self.fig)
