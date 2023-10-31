# main.py
import os
import SLither
from Ai.AI import SnakeGameAI


def main():
    model_directory = 'C:\\Users\\jhota\\PycharmProjects\\Slither AI'
    model_path = os.path.join(model_directory, 'my_model.keras')

    # Create the directory if it doesn't exist
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Initialize the SnakeGameAI with the model path
    game_ai = SnakeGameAI(model_path=model_path, training_mode=True)
    SLither.game_loop(use_ai=True, ai_agent=game_ai)
    # Train the model or start the game
    if game_ai.training_mode:
        game_ai.train(num_episodes=1000)
    else:
        # Your code to run the game normally
        pass


if __name__ == "__main__":
    main()
