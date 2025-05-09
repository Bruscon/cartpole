import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import math
import os
import pygame

def demonstrate_trained_agent():
    # Load the trained model
    model = load_model('saved_models/original_reward_function_model.keras')
    
    # Set environment variable to ensure the rendering window has a title we can find
    os.environ['SDL_VIDEO_WINDOW_POS'] = '50,50'
    
    # Create environment with rendering
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Get environment information
    state_size = env.observation_space.shape[0]
    
    # Constants for reward calculation
    MAX_ANGLE = 0.2095  # ~12 degrees threshold in radians 
    MAX_VELOCITY = 2.0  # Reasonable max velocity threshold
    
    # Initialize pygame
    pygame.init()
    font = pygame.font.SysFont('Arial', 20)
    
    # Wait for gym to create its window
    env.reset()
    time.sleep(0.5)  # Give time for window to appear
    
    # Create a transparent overlay surface
    info_overlay = pygame.Surface((600, 100), pygame.SRCALPHA)
    
    # Find the gym window to overlay our text
    gym_window = None
    for i in range(len(pygame.display.get_window_size())):
        try:
            size = pygame.display.get_window_size()[i]
            if size:
                gym_window = pygame.display.get_surface()
                break
        except:
            pass
    
    while True:  # Loop indefinitely to show multiple runs
        # Reset environment
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        score = 0
        
        # Single episode loop
        while not done:
            # Poll for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    return
            
            # Get Q-values and select best action (no epsilon, pure exploitation)
            q_values = model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Extract values for custom reward
            cart_position = next_state[0]
            cart_velocity = next_state[1]
            pole_angle = next_state[2]
            pole_angular_velocity = next_state[3]
            
            # Calculate custom reward components
            angle_reward = 1.0 - (abs(pole_angle) / MAX_ANGLE)
            velocity_reward = max(0, 1.0 - (abs(cart_velocity) / MAX_VELOCITY))
            
            # Combined reward (weighted 70% angle, 30% velocity)
            custom_reward = 0.7 * angle_reward + 0.3 * velocity_reward
            
            # Create a small text overlay
            # Print reward info to console instead
            print(f"\rAngle: {pole_angle:.4f} rad → Reward: {angle_reward:.2f} | "
                  f"Velocity: {cart_velocity:.4f} → Reward: {velocity_reward:.2f} | "
                  f"Combined: {custom_reward:.2f} | Score: {score}", end="")
            
            # Reshape state for model input
            next_state = np.reshape(next_state, [1, state_size])
            
            # Move to next state
            state = next_state
            score += 1
            
            # Add a small delay to make visualization smoother
            time.sleep(0.01)
            
            if done:
                print(f"\nEpisode ended with score: {score}")
                time.sleep(1)  # Brief pause between episodes
                break

if __name__ == "__main__":
    print("Demonstrating trained CartPole agent with custom reward visualization.")
    print("Press Ctrl+C to stop or close the window.")
    try:
        demonstrate_trained_agent()
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed properly
        try:
            pygame.quit()
        except:
            pass
        try:
            gym.envs.registry.clear()
        except:
            pass            