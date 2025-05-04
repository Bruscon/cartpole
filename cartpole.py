import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from collections import deque
import random
import time

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#use mixed precision to encourage GPU usage
tf.keras.mixed_precision.set_global_policy('mixed_float16')

#allows GPU memory to be allocated as needed
tf.config.experimental.set_memory_growth(physical_devices[0], True)

verbose = 1

# Check for GPU
if verbose > 1:
	print("TensorFlow version:", tf.__version__)
	print("GPU Available:", tf.config.list_physical_devices('GPU'))
	print("Using GPU:", tf.config.list_physical_devices('GPU'))
	print("CUDA built:", tf.test.is_built_with_cuda())


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.memory = deque(maxlen=20000)  # Experience replay buffer - increased size
        self.gamma = 0.99                  # Discount factor
        self.epsilon = 1.0                 # Exploration rate
        self.epsilon_min = 0.01            # Minimum exploration probability
        self.epsilon_decay = 0.995         # Exponential decay rate for exploration
        self.learning_rate = 0.001         # Learning rate
        self.batch_size = 512              # Size of batches for training
        self.train_start = 1000            # Minimum experiences before training
        self.update_target_frequency = 10  # How often to update target network (episodes)
        
        # Main model - trained every step
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
    
    def act(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Get Q-values for all actions in current state
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the network using randomly sampled experiences"""
        # Check if we have enough experiences
        if len(self.memory) < self.train_start:
            return
        
        # Sample a random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract batches
        states = np.array([experience[0][0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Calculate target Q values
        targets = self.model.predict(states, verbose=0)
        
        # Calculate target Q value for action using target model
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Double DQN: use main model to select action, target model to get Q-value
        max_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        
        # Update target for actions taken
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                # DDQN update
                targets[i][actions[i]] = rewards[i] + self.gamma * target_q_values[i][max_actions[i]]
        
        # Train the network
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    # Create environments - one for training, one for rendering
    env = gym.make('CartPole-v1')
    render_env = gym.make('CartPole-v1', render_mode='human')
    
    # Get environment information
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size, action_size)
    
    # Training parameters
    EPISODES = 500
    
    # Keep track of rewards per episode
    scores = []
    avg_scores = []
    
    for e in range(EPISODES):
        # Determine if we should render this episode
        if e % 50 == 0:
            current_env = render_env
        else:
            current_env = env
        
        # Reset environment
        state, _ = current_env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = current_env.step(action)
            done = terminated or truncated
            reward = reward if not done or score >= 499 else -10  # Penalty for early termination
            
            next_state = np.reshape(next_state, [1, state_size])
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            
            # Accumulate score
            score += 1
            
            # Train on past experiences (replay)
            agent.replay()
            
            if done:
                break
        
        # Update target model periodically
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()
            print(f"Episode: {e}/{EPISODES}, Target model updated")
        
        # Save scores
        scores.append(score)
        
        # Calculate average score of last 100 episodes
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        print(f"Episode: {e}/{EPISODES}, Score: {score}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # If we've solved the environment (avg score >= 195 over 100 episodes), stop
        if len(scores) >= 100 and avg_score >= 195:
            print(f"Environment solved in {e} episodes! Average score: {avg_score:.2f}")
            break
    
    agent.model.save('trained_cartpole_model.keras')

    # Close environments
    env.close()
    render_env.close()

if __name__ == "__main__":
    main()