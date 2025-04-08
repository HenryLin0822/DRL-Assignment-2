import math
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import defaultdict

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

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
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
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
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
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action, spawn = True):
        assert self.action_space.contains(action), "Invalid action"

        prev_score = self.score

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

        self.last_move_valid = moved

        if moved and spawn:
            self.add_random_tile()

        done = self.is_game_over()
        
        # Calculate immediate reward as change in score
        reward = self.score - prev_score

        return self.board, self.score, done, {"reward": reward}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
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
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
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


import math
import numpy as np
import random
import time
from collections import defaultdict

# Global approximator to avoid recreating it on every get_action call
_GLOBAL_APPROXIMATOR = None

# MCTS Node classes
class StateNode:
    """Represents a state node (max node) in the MCTS tree."""
    def __init__(self, board, parent=None):
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # action -> AfterStateNode
        self.visits = 0
        self.value = 0.0
        
    def is_fully_expanded(self, legal_actions):
        """Check if all legal actions have been tried."""
        return all(action in self.children for action in legal_actions)
    
    def select_child(self, legal_actions, exploration_weight=1.0):
        """Select child using UCB formula."""
        # If not fully expanded, return unexpanded action
        for action in legal_actions:
            if action not in self.children:
                return action
                
        # Otherwise use UCB to select
        log_n_visits = math.log(self.visits)
        
        def ucb_score(action):
            child = self.children[action]
            exploitation = child.value
            exploration = exploration_weight * math.sqrt(log_n_visits / child.visits)
            return exploitation + exploration
            
        return max(legal_actions, key=ucb_score)
        
    def add_child(self, action, afterstate_node):
        """Add a new afterstate child node."""
        self.children[action] = afterstate_node


class AfterStateNode:
    """Represents an afterstate node (chance node) in the MCTS tree."""
    def __init__(self, board, parent=None, reward=0):
        self.board = board.copy()
        self.parent = parent
        self.children = {}  # (row, col, value) -> StateNode
        self.visits = 0
        self.value = 0.0
        self.reward = reward  # Immediate reward from the action that led to this afterstate
        
    def is_fully_expanded(self, empty_cells):
        """Check if all possible random tile placements have been tried."""
        # For 2048, we need both value 2 and 4 for each empty cell
        return len(self.children) == len(empty_cells) * 2
        
    def select_child(self):
        """Select a random child based on actual 2048 probabilities."""
        # Get all possible tile placements
        placements = list(self.children.keys())
        
        # Group by position
        positions = defaultdict(list)
        for pos in placements:
            row, col, value = pos
            positions[(row, col)].append((pos, value))
            
        # First randomly select a position
        pos = random.choice(list(positions.keys()))
        
        # Then select a value based on 2048 probabilities (90% for 2, 10% for 4)
        pos_placements = positions[pos]
        if len(pos_placements) == 1:
            return pos_placements[0][0]
            
        # Find which placement has value 2 and which has value 4
        placement_2 = None
        placement_4 = None
        for placement, value in pos_placements:
            if value == 2:
                placement_2 = placement
            else:
                placement_4 = placement
                
        # Choose according to 2048 probabilities
        if random.random() < 0.9:
            return placement_2
        else:
            return placement_4
        
    def add_child(self, position, value, state_node):
        """Add a new state child node."""
        self.children[(position[0], position[1], value)] = state_node


# MCTS implementation
class MCTS:
    """Monte Carlo Tree Search implementation for 2048 with n-tuple network."""
    def __init__(self, approximator, num_iterations=1000, exploration_weight=1.0, v_norm=4096):
        self.approximator = approximator
        self.num_iterations = num_iterations
        self.exploration_weight = exploration_weight
        self.v_norm = v_norm  # Normalization constant for value scaling
        
    def choose_action(self, board, env):
        """Run MCTS and return the best action."""
        # Create root node
        root = StateNode(board)
        
        # Get legal actions
        legal_actions = [a for a in range(4) if env.is_move_legal(a)]
        
        # No legal moves
        if not legal_actions:
            return 0
            
        # Run MCTS iterations
        for _ in range(self.num_iterations):
            # Phase 1: Selection
            leaf, path = self._select(root, env, legal_actions)
            
            # Phase 2: Expansion
            if leaf.visits > 0:
                leaf = self._expand(leaf, env, path[-1][0] if path else None)
                
            # Phase 3: Simulation (using n-tuple network)
            value = self._simulate(leaf)
                
            # Phase 4: Backpropagation
            self._backpropagate(leaf, value, path)
            
        # Choose the action with the highest average value
        return self._best_action(root, legal_actions)
        
    def _select(self, node, env, legal_actions=None):
        """Select a leaf node to expand."""
        path = []  # Store (node, action) pairs along the path
        
        # If node is a state node
        while isinstance(node, StateNode):
            if legal_actions is None:
                # Get legal actions for this state
                temp_env = self._create_env(node.board)
                legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
                
            # If not fully expanded, select an unexpanded action
            if not node.is_fully_expanded(legal_actions):
                action = node.select_child(legal_actions, self.exploration_weight)
                # If this action hasn't been tried yet, we'll expand it in the expand phase
                if action not in node.children:
                    return node, path
                # Otherwise continue down the tree
                path.append((node, action))
                node = node.children[action]
            else:
                # Fully expanded, select best child
                action = node.select_child(legal_actions, self.exploration_weight)
                path.append((node, action))
                node = node.children[action]
                legal_actions = None  # Reset for next state node
                
        # If node is an afterstate node
        while isinstance(node, AfterStateNode):
            # Get empty cells for this afterstate
            empty_cells = list(zip(*np.where(node.board == 0)))
            
            # If no empty cells or fully expanded, return this node
            if not empty_cells or node.is_fully_expanded(empty_cells):
                return node, path
                
            # Otherwise select a random child to expand
            return node, path
            
        return node, path
        
    def _expand(self, node, env, previous_action=None):
        """Expand the selected node."""
        # If node is a state node, expand with an afterstate
        if isinstance(node, StateNode):
            # Get legal actions
            temp_env = self._create_env(node.board)
            legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
            
            # Select an unexpanded action
            for action in legal_actions:
                if action not in node.children:
                    # Simulate the action to get the afterstate
                    sim_env = self._create_env(node.board)
                    prev_score = sim_env.score
                    sim_env.step(action, spawn=False)  # Don't add random tile yet
                    reward = sim_env.score - prev_score
                    
                    # Create afterstate node
                    afterstate = AfterStateNode(sim_env.board, node, reward)
                    
                    # Add child node
                    node.add_child(action, afterstate)
                    return afterstate
        
        # If node is an afterstate node, expand with a state node
        elif isinstance(node, AfterStateNode):
            # Get empty cells
            empty_cells = list(zip(*np.where(node.board == 0)))
            
            # Try different tile placements
            for row, col in empty_cells:
                for value in [2, 4]:
                    key = (row, col, value)
                    if key not in node.children:
                        # Create new board with tile placed
                        new_board = node.board.copy()
                        new_board[row, col] = value
                        
                        # Create state node
                        state_node = StateNode(new_board, node)
                        
                        # Add child node
                        node.add_child((row, col), value, state_node)
                        return state_node
        
        # If we can't expand, return the node itself
        return node
        
    def _simulate(self, node):
        """Simulate using the n-tuple network instead of random playouts."""
        board = node.board
        
        # Calculate value directly using n-tuple network
        afterstate_value = self.approximator.value(board)
        
        # If it's an afterstate node, include immediate reward
        if isinstance(node, AfterStateNode):
            return (node.reward + afterstate_value) / self.v_norm
        
        # If it's a state node, we need to calculate the max afterstate value
        # by expanding all possible actions
        else:
            temp_env = self._create_env(board)
            legal_actions = [a for a in range(4) if temp_env.is_move_legal(a)]
            
            if not legal_actions:
                return 0  # Game over
                
            max_value = float('-inf')
            for action in legal_actions:
                sim_env = self._create_env(board)
                prev_score = sim_env.score
                sim_env.step(action, spawn=False)
                reward = sim_env.score - prev_score
                
                afterstate_value = self.approximator.value(sim_env.board)
                action_value = (reward + afterstate_value) / self.v_norm
                max_value = max(max_value, action_value)
                
            return max_value
        
    def _backpropagate(self, node, value, path):
        """Update statistics of visited nodes."""
        # Update the expanded node
        node.visits += 1
        node.value += (value - node.value) / node.visits
        
        # Traverse back up the path
        for parent, action in reversed(path):
            child = parent.children[action]
            value = child.value  # Use the updated value from child
            
            parent.visits += 1
            parent.value += (value - parent.value) / parent.visits
            
    def _best_action(self, root, legal_actions):
        """Select the best action based on visit count."""
        # Choose action with highest average value
        def score(action):
            if action in root.children:
                return root.children[action].visits  # Using visit count for robustness
            return 0
            
        return max(legal_actions, key=score)
        
    def _create_env(self, board):
        """Create a temporary environment with the given board state."""
        env = Game2048Env()
        env.board = board.copy()
        return env


def get_action(state, score):
    """
    Uses Monte Carlo Tree Search with the n-tuple network to select the best action.
    
    Args:
        state: The current board state (4x4 numpy array)
        score: The current score
        
    Returns:
        int: The best action (0: up, 1: down, 2: left, 3: right)
    """
    global _GLOBAL_APPROXIMATOR
    
    # Initialize the approximator if it's not already loaded
    if _GLOBAL_APPROXIMATOR is None:
        # Define the same patterns used during training
        patterns = [
            # row
            [(0,0), (0,1), (0,2), (0,3)],
            [(1,0), (1,1), (1,2), (1,3)],
            # 2*2
            [(0,0),(0,1),(1,0),(1,1)],
            [(1,0),(1,1),(2,0),(2,1)],
            # 6-tuple patterns
            [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1)],
            [(0,0), (0,1), (1,0), (1,1), (1,2), (1,3)],
            [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)],
            [(0,1), (0,2), (1,1), (1,2), (2,1), (2,2)],
            [(0,0), (0,1), (0,2), (0,3), (1,0), (1,2)],
        ]
        
        # Create and load the approximator
        from student_agent import SymmetricNTupleApproximator
        _GLOBAL_APPROXIMATOR = SymmetricNTupleApproximator(board_size=4, patterns=patterns)
        _GLOBAL_APPROXIMATOR.load('ntuple_weights_test.pkl')
    
    # Create a temporary environment to check legal moves
    from student_agent import Game2048Env
    env = Game2048Env()
    env.board = state.copy()
    env.score = score
    
    # Find legal moves
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0  # No legal moves, return any action (game is over)
    
    # If there's only one legal move, return it immediately
    if len(legal_moves) == 1:
        return legal_moves[0]
    
    # Number of MCTS iterations (adjust based on performance needs)
    num_iterations = 200  # Start with 200 iterations, can be increased for better performance
    
    # Exploration weight for UCB
    exploration_weight = 0.01
    
    # Normalization constant
    v_norm = 400000
    
    # Run MCTS
    mcts = MCTS(_GLOBAL_APPROXIMATOR, num_iterations, exploration_weight, v_norm)
    best_action = mcts.choose_action(state, env)
    
    return best_action