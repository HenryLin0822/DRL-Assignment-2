import math
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

# Color maps for visualization
COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)


# Global approximator to avoid recreating it on every get_action call
_GLOBAL_APPROXIMATOR = None

class SymmetricNTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator with symmetry transformations.
        """
        self.board_size = board_size
        self.base_patterns = patterns
        # Create a weight dictionary for each pattern
        self.weights = [defaultdict(float) for _ in patterns]
        
        # Define all symmetry transformations
        self.transformations = [
            lambda c: c,                              # identity
            lambda c: self._rot90(c),                 # 90 degrees
            lambda c: self._rot180(c),                # 180 degrees
            lambda c: self._rot270(c),                # 270 degrees
            lambda c: self._flip_h(c),                # horizontal flip
            lambda c: self._flip_v(c),                # vertical flip
            lambda c: self._flip_d1(c),               # diagonal flip (TL-BR)
            lambda c: self._flip_d2(c),               # diagonal flip (TR-BL)
        ]
    
    # Transformation methods defined as instance methods to avoid board_size parameter
    def _rot90(self, coord):
        """Rotate coordinates 90 degrees clockwise"""
        x, y = coord
        return (y, self.board_size - 1 - x)

    def _rot180(self, coord):
        """Rotate coordinates 180 degrees"""
        x, y = coord
        return (self.board_size - 1 - x, self.board_size - 1 - y)

    def _rot270(self, coord):
        """Rotate coordinates 270 degrees clockwise"""
        x, y = coord
        return (self.board_size - 1 - y, x)

    def _flip_h(self, coord):
        """Horizontal flip"""
        x, y = coord
        return (x, self.board_size - 1 - y)

    def _flip_v(self, coord):
        """Vertical flip"""
        x, y = coord
        return (self.board_size - 1 - x, y)

    def _flip_d1(self, coord):
        """Diagonal flip (top-left to bottom-right)"""
        x, y = coord
        return (y, x)

    def _flip_d2(self, coord):
        """Diagonal flip (top-right to bottom-left)"""
        x, y = coord
        return (self.board_size - 1 - y, self.board_size - 1 - x)

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords, transform_func):
        # Extract tile values from the board based on the transformed coordinates
        feature = []
        for coord in coords:
            # Apply the transformation to the coordinate
            x, y = transform_func(coord)
            
            # Ensure coordinates are valid
            if 0 <= x < self.board_size and 0 <= y < self.board_size:
                tile_value = board[x, y]
                index = self.tile_to_index(tile_value)
                feature.append(index)
            else:
                # Handle out-of-bounds coordinates
                feature.append(0)

        return tuple(feature)

    def value(self, board):
        # Estimate the board value using all symmetric transformations
        total_value = 0.0
        num_features = 0

        # For each base pattern
        for i, pattern in enumerate(self.base_patterns):
            # For each transformation
            for transform_func in self.transformations:
                # Extract feature for this pattern with this transformation
                feature = self.get_feature(board, pattern, transform_func)
                
                # Add the weight for this feature
                total_value += self.weights[i][feature]
                num_features += 1

        # Normalize by total number of features
        total_value = total_value / num_features

        return total_value

    def load(self, filename):
        """Load weights from a file"""
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)


def get_action(state, score):
    """
    Uses a symmetric n-tuple network approximator to select the best action.
    Maintains a global approximator to avoid reloading the model each time.
    Takes into account both immediate rewards and future board value.
    
    Args:
        state: The current board state (4x4 numpy array)
        score: The current score (used to calculate immediate rewards)
        
    Returns:
        int: The best action (0: up, 1: down, 2: left, 3: right)
    """
    print("1")
    global _GLOBAL_APPROXIMATOR
    
    # Initialize the approximator if it's not already loaded
    if _GLOBAL_APPROXIMATOR is None:
        # Define the same patterns used during training
        patterns = [
            #row
            [(0,0), (0,1), (0,2), (0,3)],
            [(1,0), (1,1), (1,2), (1,3)],
            #2*2
            [(0,0),(0,1),(1,0),(1,1)],
            [(1,0),(1,1),(2,0),(2,1)],
            #6-tuple patterns
            [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
            [(0,0), (0,1), (1,0), (1,1), (1,2), (1,3)],
            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
            [(0,1), (0,2), (1,1), (1,2), (2,1), (2,2)],
            [(0,0), (0,1), (0,2), (0,3), (1,0), (1,2)],
        ]
        
        # Create and load the approximator
        _GLOBAL_APPROXIMATOR = SymmetricNTupleApproximator(board_size=4, patterns=patterns)
        _GLOBAL_APPROXIMATOR.load('ntuple_weights75000.pkl')
    
    # Create a temporary environment to check legal moves
    env = Game2048Env()
    env.board = state.copy()
    
    # Find legal moves
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0  # No legal moves, return any action (game is over)
    
    # Gamma value used in training (discount factor)
    gamma = 1.0
    
    # Evaluate each legal move
    values = []
    for action in legal_moves:
        # Simulate the move and calculate immediate reward
        board_copy = state.copy()
        immediate_reward = 0
        
        if action == 0:  # Up
            for j in range(4):
                col = board_copy[:, j].copy()
                # Compress (remove zeros)
                new_col = col[col != 0]
                new_col = np.pad(new_col, (0, 4 - len(new_col)), mode='constant')
                
                # Merge
                for i in range(3):
                    if new_col[i] != 0 and new_col[i] == new_col[i+1]:
                        new_col[i] *= 2
                        immediate_reward += new_col[i]  # Add to reward
                        new_col[i+1] = 0
                
                # Compress again
                new_col = new_col[new_col != 0]
                new_col = np.pad(new_col, (0, 4 - len(new_col)), mode='constant')
                
                # Update board
                board_copy[:, j] = new_col
                
        elif action == 1:  # Down
            for j in range(4):
                col = board_copy[:, j].copy()
                # Reverse, compress, merge, compress, reverse back
                col = col[::-1]
                new_col = col[col != 0]
                new_col = np.pad(new_col, (0, 4 - len(new_col)), mode='constant')
                
                for i in range(3):
                    if new_col[i] != 0 and new_col[i] == new_col[i+1]:
                        new_col[i] *= 2
                        immediate_reward += new_col[i]
                        new_col[i+1] = 0
                
                new_col = new_col[new_col != 0]
                new_col = np.pad(new_col, (0, 4 - len(new_col)), mode='constant')
                
                # Reverse back and update
                board_copy[:, j] = new_col[::-1]
                
        elif action == 2:  # Left
            for i in range(4):
                row = board_copy[i].copy()
                # Compress
                new_row = row[row != 0]
                new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
                
                # Merge
                for j in range(3):
                    if new_row[j] != 0 and new_row[j] == new_row[j+1]:
                        new_row[j] *= 2
                        immediate_reward += new_row[j]
                        new_row[j+1] = 0
                
                # Compress again
                new_row = new_row[new_row != 0]
                new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
                
                # Update board
                board_copy[i] = new_row
                
        elif action == 3:  # Right
            for i in range(4):
                row = board_copy[i].copy()
                # Reverse, compress, merge, compress, reverse back
                row = row[::-1]
                new_row = row[row != 0]
                new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
                
                for j in range(3):
                    if new_row[j] != 0 and new_row[j] == new_row[j+1]:
                        new_row[j] *= 2
                        immediate_reward += new_row[j]
                        new_row[j+1] = 0
                
                new_row = new_row[new_row != 0]
                new_row = np.pad(new_row, (0, 4 - len(new_row)), mode='constant')
                
                # Reverse back and update
                board_copy[i] = new_row[::-1]
        
        # Calculate future value
        future_value = _GLOBAL_APPROXIMATOR.value(board_copy)
        
        # Combine immediate reward and future value
        action_value = immediate_reward + gamma * future_value
        values.append((action_value, action))
    
    # If no valid moves, return any action (game should be over)
    if not values:
        return legal_moves[0]
    
    # Return the action with the highest value
    _, best_action = max(values)
    return best_action