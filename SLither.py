import pygame
import random
import sys

# Initialize pygame and related settings
pygame.init()
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
RED = (255, 0, 0)
WIDTH, HEIGHT = 400, 300
CELL_SIZE = 10
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake Game')
clock = pygame.time.Clock()


class Snake:
    def __init__(self):
        self.positions = [(WIDTH // 2 - (WIDTH // 2 % CELL_SIZE), HEIGHT // 2 - (HEIGHT // 2 % CELL_SIZE))]
        self.direction = pygame.K_RIGHT  # Set initial direction to right
        self.segments_to_add = 0

    def move(self):
        head_x, head_y = self.positions[0]
        if self.direction == pygame.K_UP:
            head_y -= CELL_SIZE
        elif self.direction == pygame.K_DOWN:
            head_y += CELL_SIZE
        elif self.direction == pygame.K_LEFT:
            head_x -= CELL_SIZE
        elif self.direction == pygame.K_RIGHT:
            head_x += CELL_SIZE

        new_head = (head_x, head_y)
        self.positions = [new_head] + self.positions

        if self.segments_to_add == 0:
            self.positions.pop()
        else:
            self.segments_to_add -= 1

    def grow(self):
        self.segments_to_add += 2

    def draw(self, screen):
        for segment in self.positions:
            pygame.draw.rect(screen, GREEN, pygame.Rect(segment[0], segment[1], CELL_SIZE, CELL_SIZE))

    def change_direction(self, new_direction):
        if (self.direction in [pygame.K_UP, pygame.K_DOWN] and new_direction in [pygame.K_LEFT, pygame.K_RIGHT]) or \
                (self.direction in [pygame.K_LEFT, pygame.K_RIGHT] and new_direction in [pygame.K_UP, pygame.K_DOWN]):
            self.direction = new_direction


class Food:
    def __init__(self, snake_positions):
        self.randomize_position(snake_positions)

    def randomize_position(self, snake_positions):
        while True:
            self.position = (random.randint(0, (WIDTH - CELL_SIZE) // CELL_SIZE) * CELL_SIZE,
                             random.randint(0, (HEIGHT - CELL_SIZE) // CELL_SIZE) * CELL_SIZE)
            if self.position not in snake_positions:
                break

    def draw(self, screen):
        pygame.draw.rect(screen, RED, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE))


def draw_score(score, screen):
    WHITE = (255,255, 255)
    font = pygame.font.SysFont('arial', 24)  # Using Arial font for better compatibility
    score_surface = font.render(f'Score: {score}', True, WHITE)
    score_rect = score_surface.get_rect(topright=(WIDTH - 10, 10))
    screen.blit(score_surface, score_rect)

def game_step(snake, food, direction):
    snake.change_direction(direction)
    snake.move()
    reward = 0
    game_over = False

    # Check if snake ate the food
    if pygame.Rect(snake.positions[0][0], snake.positions[0][1], CELL_SIZE, CELL_SIZE).colliderect(
            pygame.Rect(food.position[0], food.position[1], CELL_SIZE, CELL_SIZE)):
        snake.grow()
        food.randomize_position(snake.positions)
        reward = 10  # Reward for eating food

    # Check for collision with boundaries or itself
    head_collides_with_body = any(segment for segment in snake.positions[1:] if segment == snake.positions[0])
    head_collides_with_boundaries = not (0 <= snake.positions[0][0] < WIDTH and 0 <= snake.positions[0][1] < HEIGHT)
    if head_collides_with_body or head_collides_with_boundaries:
        game_over = True
        reward = -10  # Penalty for game over

    # Return game state, reward, and game status
    return get_state(snake, food), reward, game_over
def reset_game():
    snake = Snake()
    food = Food(snake.positions)
    return snake, food
def game_over_screen(score):
    font = pygame.font.SysFont(None, 55)
    text_surface = font.render('Game Over', True, RED)
    text_rect = text_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2))
    score_surface = font.render(f'Final Score: {score}', True, RED)
    score_rect = score_surface.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 60))

    screen.blit(text_surface, text_rect)
    screen.blit(score_surface, score_rect)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    return True


def game_loop(use_ai=False, ai_agent=None):
    speed = 10
    snake = Snake()
    food = Food(snake.positions)
    score = 0
    running = True

    while running:
        if use_ai:
            action = ai_agent.predict_action(snake, food)
            snake.change_direction(action)
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    snake.change_direction(event.key)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                snake.change_direction(event.key)

        snake.move()

        if pygame.Rect(snake.positions[0][0], snake.positions[0][1], CELL_SIZE, CELL_SIZE).colliderect(
                pygame.Rect(food.position[0], food.position[1], CELL_SIZE, CELL_SIZE)):
            snake.grow()
            food.randomize_position(snake.positions)
            score += 1

        head_collides_with_body = any(segment for segment in snake.positions[1:] if segment == snake.positions[0])
        head_collides_with_boundaries = not (0 <= snake.positions[0][0] < WIDTH and 0 <= snake.positions[0][1] < HEIGHT)

        if head_collides_with_body or head_collides_with_boundaries:
            running = False
            restart = game_over_screen(score)  # Get restart value
            if restart:
                game_loop()  # Restart the game
            else:
                pygame.quit()
                return  # Exit the function to avoid the rest of the code
        if not use_ai:
            font = pygame.font.SysFont('arial', 20)
            human_playing_text = font.render('Human Playing', True, GREEN)
            screen.blit(human_playing_text, (WIDTH - 150, HEIGHT - 30))
        screen.fill(BLACK)
        snake.draw(screen)
        food.draw(screen)
        draw_score(score, screen)  # Display the score
        pygame.display.flip()

        clock.tick(speed)

    pygame.quit()


