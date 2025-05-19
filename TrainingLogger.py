import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for Ubuntu
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
import time
from collections import deque

# Use TkAgg backend which works well on Ubuntu
matplotlib.use('TkAgg')

class TrainingLogger:
    def __init__(self, log_dir, window_size=25):
        self.log_dir = log_dir
        self.window_size = window_size
        
        # Create DataFrame to store all metrics
        self.data = pd.DataFrame(columns=[
            'episode_reward', 
            'episode_length',
            'eval_length',
            'loss', 
            'learning_rate', 
            'epsilon',
            'avg_reward',  # Track rolling average as a continuous series
            'avg_length'   # Track rolling average as a continuous series
        ])
        
        # Create rolling windows for episode metrics
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        
        # Setup for non-blocking updates
        self.last_update_time = time.time()
        self.update_interval = 1.0  # Update plot every 1 second
        
        # Create plot figure and axes
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure and axes"""
        plt.ion()  # Turn on interactive mode
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
        self.training_ax.set_ylim(0, 1)  # For loss and epsilon (0-1)
        
        # Create a second y-axis for learning rate
        self.lr_ax = self.training_ax.twinx()
        self.lr_ax.set_ylabel('Learning Rate')
        self.lr_ax.set_ylim(0, 0.1)  # For learning rate (0-0.1)
        
        # Show the plot
        plt.tight_layout()
        plt.show(block=False)
    
    def log_metrics(self, step, reward=None, length=None, eval_length=None, loss=None, lr=None, epsilon=None):
        """Log all metrics at once for a given step"""
        # If an episode was completed, update rolling averages
        if reward is not None:
            self.reward_window.append(reward)
        if length is not None:
            self.length_window.append(length)
            
        # Calculate current rolling averages
        avg_reward = np.mean(self.reward_window) if self.reward_window else np.nan
        avg_length = np.mean(self.length_window) if self.length_window else np.nan
        
        # Create a new row with NaN values
        new_data = pd.Series(
            {
                'episode_reward': reward, 
                'episode_length': length,
                'eval_length': eval_length,
                'loss': loss, 
                'learning_rate': lr, 
                'epsilon': epsilon,
                'avg_reward': avg_reward if not np.isnan(avg_reward) else None,
                'avg_length': avg_length if not np.isnan(avg_length) else None
            }, 
            name=step
        )
        
        # If this step already exists, update non-None values
        if step in self.data.index:
            for column in self.data.columns:
                value = new_data[column]
                if pd.notna(value):  # Only update if the new value is not NaN
                    self.data.at[step, column] = value
        else:
            # Otherwise add the new row
            self.data = pd.concat([self.data, pd.DataFrame([new_data])])
            
        # Sort by index (step)
        self.data = self.data.sort_index()
    
    def update_plot(self, save=False):
        """Update the plot with current data"""
        if self.data.empty:
            return
            
        # Clear previous plots
        self.episode_ax.clear()
        self.training_ax.clear()
        self.lr_ax.clear()
        
        # Reset titles and limits
        self.episode_ax.set_title('Episode Metrics')
        self.episode_ax.set_ylabel('Episode Length / Reward')
        self.episode_ax.set_ylim(0, 500)
        
        self.training_ax.set_title('Training Metrics')
        self.training_ax.set_xlabel('Step')
        self.training_ax.set_ylabel('Loss / Epsilon')
        self.training_ax.set_ylim(0, 1)  # For loss and epsilon (0-1)
        
        self.lr_ax.set_ylabel('Learning Rate')
        self.lr_ax.set_ylim(0, 0.1)  # For learning rate (0-0.1)
        
        # Drop NaN values for each metric
        reward_data = self.data['episode_reward'].dropna()
        length_data = self.data['episode_length'].dropna()
        eval_length_data = self.data['eval_length'].dropna()
        avg_reward_data = self.data['avg_reward'].dropna()
        avg_length_data = self.data['avg_length'].dropna()
        loss_data = self.data['loss'].dropna()
        lr_data = self.data['learning_rate'].dropna()
        epsilon_data = self.data['epsilon'].dropna()
        
        # ===== Plot episode metrics =====
        # Scatter plots for individual episodes
        if not reward_data.empty:
            self.episode_ax.scatter(reward_data.index, reward_data.values, 
                                   color='blue', alpha=0.5, s=10, label='Episode Reward')
        
        if not length_data.empty:
            self.episode_ax.scatter(length_data.index, length_data.values, 
                                   color='green', alpha=0.5, s=10, label='Episode Length')
        
        # Line plot for evaluation length
        if not eval_length_data.empty:
            self.episode_ax.plot(eval_length_data.index, eval_length_data.values, 
                                'r-', label='Eval Length')
        
        # Line plots for rolling averages
        if not avg_reward_data.empty:
            self.episode_ax.plot(avg_reward_data.index, avg_reward_data.values, 
                                'b-', linewidth=2, label=f'Avg Reward ({self.window_size} ep)')
        
        if not avg_length_data.empty:
            self.episode_ax.plot(avg_length_data.index, avg_length_data.values, 
                                'g-', linewidth=2, label=f'Avg Length ({self.window_size} ep)')
        
        # ===== Plot training metrics =====
        # Plot loss and epsilon on left axis
        if not loss_data.empty:
            self.training_ax.plot(loss_data.index, loss_data.values, 
                                 'r-', label='Loss')
        
        if not epsilon_data.empty:
            self.training_ax.plot(epsilon_data.index, epsilon_data.values, 
                                 'g-', label='Epsilon')
        
        # Plot learning rate on right axis
        if not lr_data.empty:
            self.lr_ax.plot(lr_data.index, lr_data.values, 
                           'b-', label='Learning Rate')
        
        # Add legends
        self.episode_ax.legend(loc='upper left')
        handles1, labels1 = self.training_ax.get_legend_handles_labels()
        handles2, labels2 = self.lr_ax.get_legend_handles_labels()
        self.training_ax.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        
        # Update layout
        plt.tight_layout()
        
        # Refresh plot
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # Allow GUI to process events
        except Exception as e:
            print(f"Warning: Could not update plot interactively: {e}")
        
        # Save plot if requested
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