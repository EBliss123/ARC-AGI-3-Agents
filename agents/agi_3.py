# All AGI components in a single file
import numpy as np


# -----------------
# 1. Memory System
# -----------------
class AgentMemory:
    """
    Stores the AGI's memory.
    [cite_start]This includes seen grid states to aid in curiosity-driven exploration[cite: 39].
    """
    def __init__(self):
        self.seen_states = set()
        print("Memory System online.")

    def remember_state(self, grid_state):
        # We'll need a way to make the grid 'hashable' to store it in a set
        self.seen_states.add(grid_state.tobytes())

    def has_seen_state(self, grid_state):
        return grid_state.tobytes() in self.seen_states

# -----------------
# 2. Perception System
# -----------------
class PerceptionSystem:
    """
    Receives raw game data and creates a structured model.
    Handles Object Segmentation by grouping pixels and defining object attributes. [cite: 32, 33]
    """
    def __init__(self):
        print("Perception System online.")

    def _bfs_search(self, start_node, grid, visited):
        """
        Performs a Breadth-First Search to find all connected pixels of the same color.
        """
        q = [start_node]
        visited.add(start_node)
        
        object_pixels = []
        color = grid[start_node[0]][start_node[1]]

        while q:
            row, col = q.pop(0)
            object_pixels.append((row, col))

            # Check neighbors (up, down, left, right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc

                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) not in visited and grid[nr][nc] == color:
                        visited.add((nr, nc))
                        q.append((nr, nc))
        
        return object_pixels, color

    def process_grid(self, grid):
        """
        Processes the raw grid to identify objects by scanning and grouping pixels. [cite: 33]
        """
        visited = set()
        objects = []
        object_id_counter = 1

        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if (r, c) not in visited: 
                    
                    # Found a new object, start a search
                    pixels, color = self._bfs_search((r, c), grid, visited)
                    
                    # Record its attributes [cite: 34]
                    min_r = min(p[0] for p in pixels)
                    max_r = max(p[0] for p in pixels)
                    min_c = min(p[1] for p in pixels)
                    max_c = max(p[1] for p in pixels)
                    
                    obj = {
                        "id": object_id_counter,
                        "color": int(color),
                        "position": (min_c, min_r, max_c - min_c + 1, max_r - min_r + 1), # x, y, width, height
                        "size": len(pixels),
                        "shape": sorted(pixels)
                    }
                    objects.append(obj)
                    object_id_counter += 1
        
        print(f"Perception complete. Found {len(objects)} objects.")
        return objects

# -----------------
# 3. Learning System
# -----------------
class LearningAgent:
    """
    Contains the core logic for learning and decision making.
    [cite_start]Manages action discovery, rule synthesis, and exploration[cite: 36, 38, 39].
    """
    def __init__(self):
        print("Learning Agent online.")

    def discover_actions(self, previous_grid, new_grid, action):
        """
        [cite_start]Compares grids before and after an action to see what changed[cite: 36, 37].
        """
        print(f"Discovering effect of action: {action}")

# -----------------
# 4. Main Agent Logic
# -----------------
def run_agent():
    """
    Main function to run the AGI agent.
    """
    # Initialize the core components
    agent_memory = AgentMemory()
    perception_system = PerceptionSystem()
    learning_agent = LearningAgent()

    print("\nAGI Agent Initialized. Ready to solve level 1.")

    # Create a sample grid to test perception
    # 0 = background, 1 = red, 2 = blue
    test_grid = np.array([
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 2, 0],
        [0, 3, 3, 3, 0, 0, 0],
    ])

    print("\n--- Running Perception Test ---")
    # Process the grid to find objects
    identified_objects = perception_system.process_grid(test_grid)

    # Print the results
    import json
    print(json.dumps(identified_objects, indent=2))
    print("-----------------------------\n")

if __name__ == "__main__":
    run_agent()