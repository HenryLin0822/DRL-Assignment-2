# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


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

def get_action(state, score):
    """
    Uses an n-tuple network approximator to select the best action for the current game state.
    
    Args:
        state: The current board state (4x4 numpy array)
        score: The current score (unused in this implementation)
        
    Returns:
        int: The best action (0: up, 1: down, 2: left, 3: right)
    """
    import math
    import numpy as np
    import pickle
    import copy
    import sys
    
    # Fix numpy._core module issue for pickle loading
    if not hasattr(np, '_core'):
        np._core = np.core
        sys.modules['numpy._core'] = np.core
    
    # Define the N-Tuple Approximator class
    class NTupleApproximator:
        def __init__(self, board_size, patterns):
            """
            Initializes the N-Tuple approximator without symmetry transformations.
            """
            self.board_size = board_size
            self.patterns = patterns
            # Create a weight dictionary for each pattern
            self.weights = [{} for _ in patterns]
            
        def tile_to_index(self, tile):
            """
            Converts tile values to an index for the lookup table.
            """
            if tile == 0:
                return 0
            else:
                return int(math.log(tile, 2))
                
        def get_feature(self, board, coords):
            """
            Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
            """
            feature = []
            for x, y in coords:
                # Ensure coordinates are valid
                if 0 <= x < self.board_size and 0 <= y < self.board_size:
                    tile_value = board[x, y]
                    index = self.tile_to_index(tile_value)
                    feature.append(index)
                else:
                    feature.append(0)
                    
            return tuple(feature)
            
        def value(self, board):
            """
            Estimate the value of a board position.
            """
            total_value = 0.0
            
            # Sum values from all patterns
            for i, pattern in enumerate(self.patterns):
                feature = self.get_feature(board, pattern)
                
                # Get weight for this feature
                if feature in self.weights[i]:
                    total_value += self.weights[i][feature]
                    
            return total_value
            
        def load(self, weights_data):
            """Load weights from provided data"""
            self.weights = weights_data
    
    # Define the patterns to use (same as in the training code)
    patterns = [
        # All rows
        [(0,0), (0,1), (0,2), (0,3)],  # Row 0
        [(1,0), (1,1), (1,2), (1,3)],  # Row 1
        [(2,0), (2,1), (2,2), (2,3)],  # Row 2
        [(3,0), (3,1), (3,2), (3,3)],  # Row 3

        # All columns
        [(0,0), (1,0), (2,0), (3,0)],  # Column 0
        [(0,1), (1,1), (2,1), (3,1)],  # Column 1
        [(0,2), (1,2), (2,2), (3,2)],  # Column 2
        [(0,3), (1,3), (2,3), (3,3)],  # Column 3

        # All 2×2 squares
        [(0,0), (0,1), (1,0), (1,1)],  # Top-left
        [(0,1), (0,2), (1,1), (1,2)],  # Top-middle-left
        [(0,2), (0,3), (1,2), (1,3)],  # Top-middle-right
        [(1,0), (1,1), (2,0), (2,1)],  # Middle-left
        [(1,1), (1,2), (2,1), (2,2)],  # Middle-center
        [(1,2), (1,3), (2,2), (2,3)],  # Middle-right
        [(2,0), (2,1), (3,0), (3,1)],  # Bottom-left
        [(2,1), (2,2), (3,1), (3,2)],  # Bottom-middle-left
        [(2,2), (2,3), (3,2), (3,3)]   # Bottom-right
    ]
    
    # Create an approximator instance
    approximator = NTupleApproximator(board_size=4, patterns=patterns)
    
    # Load the weights safely
    try:
        with open('improved_ntuple_weights.pkl', 'rb') as f:
            weights_data = pickle.load(f)
        approximator.load(weights_data)
    except Exception as e:
        # If loading fails, try with encoding parameter
        try:
            with open('improved_ntuple_weights.pkl', 'rb') as f:
                weights_data = pickle.load(f, encoding='latin1')
            approximator.load(weights_data)
        except Exception as e2:
            # If still failing, use random action as last resort
            print(f"Error loading model: {e2}")
            import random
            return random.choice([0, 1, 2, 3])
    
    # Find legal moves
    legal_moves = []
    for action in range(4):
        # Create a copy of the game to test if move is legal
        test_board = state.copy()
        env = Game2048Env()
        env.board = test_board
        if env.is_move_legal(action):
            legal_moves.append(action)
    
    if not legal_moves:
        # No legal moves, return any action
        return 0
    
    # Use after-states for action selection
    values = []
    for action in legal_moves:
        # Create a copy of the environment to simulate the action
        env = Game2048Env()
        env.board = state.copy()
        
        # Execute action without spawning a random tile
        if action == 0:
            moved = env.move_up()
        elif action == 1:
            moved = env.move_down()
        elif action == 2:
            moved = env.move_left()
        elif action == 3:
            moved = env.move_right()
        
        # Skip invalid moves
        if not moved:
            continue
            
        after_state = env.board.copy()
        
        # Get the value estimation for the resulting after-state
        state_value = approximator.value(after_state)
        values.append((state_value, action))
    
    if not values:
        # If no moves were evaluated (shouldn't happen if legal_moves is correct)
        return legal_moves[0]
        
    # Choose the action with the highest estimated value
    values.sort(reverse=True)
    best_value, best_action = values[0]
    
    return best_action