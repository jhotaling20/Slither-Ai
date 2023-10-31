import math

import pygame
import numpy as np
import os
from collections import deque
import random
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.src.layers import Dropout
from pygame.time import Clock

# Assuming SLither.py contains the definitions for your Snake, Food classes and related parameters
from SLither import Snake, Food, WIDTH, HEIGHT, CELL_SIZE


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


def turn_left(current_direction):
    if current_direction == pygame.K_UP:
        return pygame.K_LEFT
    elif current_direction == pygame.K_DOWN:
        return pygame.K_RIGHT
    elif current_direction == pygame.K_LEFT:
        return pygame.K_DOWN
    elif current_direction == pygame.K_RIGHT:
        return pygame.K_UP
    return current_direction


class SnakeGameAI:
    def __init__(self, model_path='my_model.keras', training_mode=True, memory_size=1000, batch_size=50):
        self.score = 0
        pygame.init()
        self.render_every = 50  # Render every 50 iterations, adjust as needed
        self.iteration = 0
        self.steps_in_current_game = 0
        self.best_score = 0
        self.gamma = 0.95  # or another value you find appropriate
        self.game_reward = 0
        self.num_actions = 4  # adjust based on your action size
        self.model_path = model_path
        self.training_mode = training_mode
        self.memory = Memory(max_size=memory_size)
        self.batch_size = batch_size
        self.reward = 0
        # Game parameters
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.WIDTH, self.HEIGHT = 400, 300
        self.CELL_SIZE = 20
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.DOUBLEBUF)
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        # AI parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.decay_rate = 0.995
        self.model = self.load_model()

        # For displaying information
        self.font = pygame.font.SysFont('arial', 25)
        self.episodes = 0

    def load_model(self):
        if os.path.isfile(self.model_path):
            print("Loading existing model")
            return load_model(self.model_path)
        else:
            print("Creating new model")
            return self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(4,)))  # Updated input_shape to 4
        model.add(Dropout(0.1))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.0005))
        return model

    def save_model(self):
        self.model.save(self.model_path)

    def reset(self):
        self.snake = Snake()
        self.food = Food(self.snake.positions)
        self.score = 0
        self.game_reward = 0
        return self.get_state()

    def get_state(self):
        head_x, head_y = self.snake.positions[0]
        food_x, food_y = self.food.position

        # Calculate angle to the food
        angle_to_food = self.calculate_angle_to_food(head_x, head_y, food_x, food_y)

        # Calculate distance to the nearest wall
        distance_to_wall = self.calculate_distance_to_wall(head_x, head_y)

        state = [
            head_x - food_x,  # X distance to food
            head_y - food_y,  # Y distance to food
            angle_to_food,  # Angle to the food
            distance_to_wall  # Distance to the nearest wall
        ]
        return np.array(state)

    def calculate_angle_to_food(self, head_x, head_y, food_x, food_y):
        # Calculate the angle between the snake's head and the food
        # You might need to adjust this calculation based on how your game defines directions
        delta_x = food_x - head_x
        delta_y = food_y - head_y
        angle = math.atan2(delta_y, delta_x)
        return angle

    def calculate_distance_to_wall(self, head_x, head_y):
        # Find the minimum distance to any of the four walls
        distance_to_left_wall = head_x
        distance_to_right_wall = self.WIDTH - head_x
        distance_to_top_wall = head_y
        distance_to_bottom_wall = self.HEIGHT - head_y
        return min(distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall)

    def step(self, action):
        # Map the numeric action to the actual game direction
        direction_mapping = {
            0: pygame.K_UP,
            1: pygame.K_DOWN,
            2: pygame.K_LEFT,
            3: pygame.K_RIGHT
        }
        game_direction = direction_mapping.get(action, pygame.K_RIGHT)  # Default to moving right

        # Calculate the distance before moving
        distance_before_move = self.calculate_distance(self.snake.positions[0], self.food.position)
        old_head_x, old_head_y = self.snake.positions[0]
        center_x, center_y = self.WIDTH // 2, self.HEIGHT // 2
        old_distance_from_center = np.sqrt((old_head_x - center_x) ** 2 + (old_head_y - center_y) ** 2)
        self.snake.change_direction(game_direction)
        self.snake.move()

        new_head_x, new_head_y = self.snake.positions[0]
        new_distance_from_center = np.sqrt((new_head_x - center_x) ** 2 + (new_head_y - center_y) ** 2)

        # Calculate the distance after moving
        distance_after_move = self.calculate_distance(self.snake.positions[0], self.food.position)
        self.game_reward = 0
        game_over = False
        immediate_reward = 0  # Reward for the current action
        if self.score > self.best_score:
            self.best_score = self.score

        if self.check_collisions():
            game_over = True
            immediate_reward = -100  # Large penalty for dying
        if self.food_collided():
            self.snake.grow()
            self.food.randomize_position(self.snake.positions)
            self.score += 1
            immediate_reward = 50
        elif distance_after_move < distance_before_move:
            immediate_reward = 25  # Reward for moving closer to food
        elif distance_after_move > distance_before_move:
            immediate_reward = -10  # Penalty for moving away from food

        # Update the game reward with the immediate reward
        self.game_reward += immediate_reward
        self.iteration += 1
        if self.iteration % self.render_every == 0:
            self.render()
        return self.get_state(), immediate_reward, game_over

    def calculate_distance(self, pos1, pos2):
        # Manhattan distance
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def check_collisions(self):
        head = self.snake.positions[0]
        return head[0] < 0 or head[0] >= WIDTH or head[1] < 0 or head[1] >= HEIGHT or head in self.snake.positions[1:]

    def food_collided(self):
        head = self.snake.positions[0]
        return head == self.food.position

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            self.episodes += 1
            state = self.reset()
            game_over = False
            while not game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                if np.random.rand() <= self.epsilon:
                    action = np.random.randint(0, 4)
                else:
                    q_values = self.model.predict(state.reshape((1, -1)))
                    action = np.argmax(q_values[0])

                next_state, reward, game_over = self.step(action)

                self.memory.add((state, action, reward, next_state, game_over))
                state = next_state

                if len(self.memory.buffer) > self.batch_size:
                    minibatch = self.memory.sample(self.batch_size)
                    for state, action, reward, next_state, done in minibatch:
                        target = reward
                        if not done:
                            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, -1)))[0])
                        target_f = self.model.predict(state.reshape((1, -1)))
                        target_f[0][action] = target
                        self.model.fit(state.reshape((1, -1)), target_f, epochs=1, verbose=0)

                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                else:
                    self.epsilon = self.epsilon_min

                if not self.training_mode:
                    self.render()
            self.reward += self.game_reward
            state = self.reset()
            if episode % 10 == 0:
                self.save_model()

    def render(self):
        self.screen.fill(self.BLACK)
        for pos in self.snake.positions:
            pygame.draw.rect(self.screen, self.GREEN, [pos[0], pos[1], CELL_SIZE, CELL_SIZE])
        pygame.draw.rect(self.screen, self.RED, [self.food.position[0], self.food.position[1], CELL_SIZE, CELL_SIZE])

        score_text = self.font.render(f'Score: {self.score}', True, self.GREEN)
        self.screen.blit(score_text, [0, 0])
        episode_text = self.font.render(f'Episode: {self.episodes}', True, self.GREEN)
        self.screen.blit(episode_text, [0, 25])
        epsilon_text = self.font.render(f'Epsilon: {self.epsilon:.2f}', True, self.GREEN)
        self.screen.blit(epsilon_text, [0, 50])
        reward_text = self.font.render(f'Reward: {self.reward}', True, self.GREEN)  # New line to render the reward
        self.screen.blit(reward_text, [0, 75])
        game_award_txt = self.font.render(f'Game Reward: {self.game_reward}', True,
                                          self.GREEN)  # New line to render the reward
        self.screen.blit(game_award_txt, [0, 100])
        High_Score_text = self.font.render(f'High Score: {self.best_score}', True,
                                           self.GREEN)  # New line to render the reward
        self.screen.blit(High_Score_text, [0, 125])

        pygame.display.flip()
        self.clock.tick(10)


# Example usage
game_ai = SnakeGameAI(training_mode=True)

# Train the model
game_ai.train()
