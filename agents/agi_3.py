class GameObject:
    def __init__(self, obj_id, color, size, position, shape):
        self.id = obj_id
        self.color = color
        self.size = size
        self.position = position
        self.shape = shape

    def __repr__(self):
        return f"ID: {self.id}, Color: {self.color}, Size: {self.size}, Pos: {self.position}"
    
def segment_grid(grid):
    # Get the dimensions of the grid
    
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    
    # Create a grid to track visited pixels
    visited = [[False for _ in range(width)] for _ in range(height)]
    
    found_objects = []
    object_id = 1

    # Helper function to find all pixels of an object using BFS
    def _find_object_pixels(start_r, start_c):
        q = [(start_r, start_c)]  # A queue for pixels to visit
        pixels = []  # The list of pixels in this object
        target_color = grid[start_r][start_c]
        
        # Mark the starting pixel as visited
        visited[start_r][start_c] = True
        
        while q:
            r, c = q.pop(0)
            pixels.append((r, c))
            
            # Check neighbors (up, down, left, right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                
                # Check if the neighbor is valid
                if 0 <= nr < height and 0 <= nc < width and \
                   not visited[nr][nc] and grid[nr][nc] == target_color:
                    
                    visited[nr][nc] = True
                    q.append((nr, nc))
                    
        return pixels
    
    # Iterate over every pixel to find objects
    for r in range(height):
        for c in range(width):
            # If the pixel has not been visited, it could be a new object
            if not visited[r][c]:
                # Found a new, unvisited pixel. Start a search.
                object_pixels = _find_object_pixels(r, c)
                
                # Now, create the GameObject
                color = grid[r][c]
                size = len(object_pixels)
                
                # Calculate bounding box and relative shape
                min_r = min(p[0] for p in object_pixels)
                min_c = min(p[1] for p in object_pixels)
                position = (min_r, min_c)
                
                shape = [(r - min_r, c - min_c) for r, c in object_pixels]

                # Create the final GameObject
                obj = GameObject(object_id, color, size, position, shape)
                found_objects.append(obj)
                object_id += 1

    return found_objects

# --- Test Code ---
import gymnasium as gym
import arc_prize_utils

if __name__ == "__main__":
    print("--- Loading ARC-AGI-3 Game Environment: LS20 ---")
    
    # This creates the interactive game environment for LS20
    # The 'v0' at the end is just a version number
    env = gym.make('arc-v2-ls20-v0')

    # Reset the environment to get the starting screen (grid)
    # observation is the grid, info is other data we don't need yet
    observation, info = env.reset()

    print("\nInitial grid received from the game environment.")
    
    # Run our segmentation function on this initial grid
    result_objects = segment_grid(observation)

    print("\nSegmentation finished. Found objects:")
    for obj in result_objects:
        print(obj)
        
    # Close the environment when done
    env.close()