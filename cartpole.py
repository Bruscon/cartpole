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
import math 

verbose = 1

# Check for GPU
if verbose > 1:
	print("TensorFlow version:", tf.__version__)
	print("GPU Available:", tf.config.list_physical_devices('GPU'))
	print("Using GPU:", tf.config.list_physical_devices('GPU'))
	print("CUDA built:", tf.test.is_built_with_cuda())


def get_discrete_state(state):
	discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
	return tuple(discrete_state.astype(np.int32))

if __name__ == "__main__":

	main_env = gym.make("CartPole-v1")  # For normal training
	render_env = gym.make("CartPole-v1", render_mode="human")  # For rendering

	LEARNING_RATE = 0.1

	DISCOUNT = 0.95
	EPISODES = 60000
	total = 0
	total_reward = 0
	prior_reward = 0

	Observation = [30, 30, 50, 50]
	np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

	epsilon = 1

	epsilon_decay_value = 0.99995

	q_table = np.random.uniform(low=0, high=1, size=(Observation + [main_env.action_space.n]))
	q_table.shape

	for episode in range(EPISODES + 1): #go through the episodes

		# Choose which environment to use this episode
		if episode % 2000 == 0:
			env = render_env  # Use rendering environment
		else:
			env = main_env  # Use non-rendering environment
			
		t0 = time.time() #set the initial time
		discrete_state = get_discrete_state(env.reset()[0]) #get the discrete start for the restarted environment 
		done = False
		episode_reward = 0 #reward starts as 0 for each episode

		if episode % 2000 == 0: 
			print("Episode: " + str(episode))

		while not done: 

			if np.random.random() > epsilon:

				action = np.argmax(q_table[discrete_state]) #take cordinated action
			else:

				action = np.random.randint(0, env.action_space.n) #do a random ation

			new_state, reward, terminated, truncated, info = env.step(action) #step action to get new states, reward, and the "done" status.

			done = terminated or truncated

			episode_reward += reward #add the reward

			new_discrete_state = get_discrete_state(new_state)


			if not done: #update q-table
				max_future_q = np.max(q_table[new_discrete_state])

				current_q = q_table[discrete_state + (action,)]

				new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

				q_table[discrete_state + (action,)] = new_q

			discrete_state = new_discrete_state

		if epsilon > 0.05: #epsilon modification
			if episode_reward > prior_reward and episode > 10000:
				epsilon = math.pow(epsilon_decay_value, episode - 10000)

				if episode % 500 == 0:
					print("Epsilon: " + str(epsilon))

		t1 = time.time() #episode has finished
		episode_total = t1 - t0 #episode total time
		total = total + episode_total

		total_reward += episode_reward #episode total reward
		prior_reward = episode_reward

		if episode % 1000 == 0: #every 1000 episodes print the average time and the average reward
			mean = total / 1000
			print("Time Average: " + str(mean))
			total = 0

			mean_reward = total_reward / 1000
			print("Mean Reward: " + str(mean_reward))
			total_reward = 0

	env.close()