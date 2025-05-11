import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

def create_initial_model(state_size=4, action_size=2, save_path="initial_model.keras"):
    """Create and save an initial model with random weights"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Build the model
    model = Sequential([
        Dense(24, input_dim=state_size, activation='relu'),
        Dense(24, activation='relu'),
        Dense(action_size, activation='linear')
    ])
    
    # Compile the model
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    # Save the model
    model.save(save_path)
    print(f"Initial model created and saved to: {save_path}")
    
    return model

if __name__ == "__main__":
    # You can customize these parameters if needed
    create_initial_model(
        state_size=4,  # CartPole observation space size
        action_size=2,  # CartPole action space size
        save_path="saved_models/initial_model.keras"  # Path where to save the model
    )