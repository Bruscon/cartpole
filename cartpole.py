import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Check for GPU
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("Using GPU:", tf.config.list_physical_devices('GPU'))
print("CUDA built:", tf.test.is_built_with_cuda())

# Try a simple operation on GPU
if len(tf.config.list_physical_devices('GPU')) > 0:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("Matrix multiplication result:", c)

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')

# Simple neural network for CartPole
model = Sequential([
	Input(shape=(4,)),
    Dense(24, activation='relu'),
    Dense(24, activation='relu'),
    Dense(2, activation='linear')
])

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

# Run a simple test episode
observation, info = env.reset()
for _ in range(200):
    # Random action for this test
    action = env.action_space.sample()
    
    # Use model for prediction (for GPU testing)
    state_tensor = tf.convert_to_tensor(observation.reshape(1, 4))
    with tf.device('/GPU:0'):
        q_values = model(state_tensor)
    
    observation, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
print("Test completed successfully!")