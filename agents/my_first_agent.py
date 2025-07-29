import numpy as np # We'll use numpy for easier grid manipulation

class MyFirstAgent:
    def __init__(self, agent_id: str):
        """
        Initializes your agent.

        Args:
            agent_id (str): A unique identifier for this agent instance.
        """
        self.id = agent_id
        print(f"MyFirstAgent '{self.id}' initialized!")

    def process_task(self, task_data: dict) -> dict:
        """
        Processes a single ARC-AGI task and returns a predicted output grid.

        Args:
            task_data (dict): A dictionary containing the task information.
                              It has two main keys:
                              - 'train': A list of dicts, each with 'input' and 'output' grids for training.
                              - 'test': A list of dicts, each with an 'input' grid for testing.
                                        (You need to predict the 'output' for these).

        Returns:
            dict: A dictionary with a single key 'output',
                  whose value is your agent's predicted output grid (list of lists of integers).
                  Example: {"output": [[0, 0], [0, 0]]}
        """
        print(f"Agent '{self.id}' is processing a new task...")

        # --- Understanding the Input (task_data) ---
        # Let's extract the first test input grid for simplicity for now.
        # task_data['test'] is a list of test cases, each with an 'input' grid.
        # We typically focus on the first test case to generate an output.
        test_input_grid = task_data['test'][0]['input']
        print(f"Test input grid dimensions: {len(test_input_grid)} rows x {len(test_input_grid[0])} columns")

        # --- Your FIRST Simple Agent Logic (Placeholder) ---
        # For this initial setup, let's create a dummy output:
        # We'll just return a black grid (all zeros) that has the same dimensions as the test input.
        
        # Convert to numpy array for easier shape access (optional but good practice)
        test_input_np = np.array(test_input_grid)
        output_rows = test_input_np.shape[0]
        output_cols = test_input_np.shape[1]

        # Create a new grid filled with zeros (black color)
        predicted_output_grid = [[0 for _ in range(output_cols)] for _ in range(output_rows)]

        print(f"Agent '{self.id}' generated a black grid of size {output_rows}x{output_cols}")

        # --- Return Your Prediction ---
        # The return format MUST be a dictionary like this: {"output": your_grid_data}
        return {"output": predicted_output_grid}

# Note: You don't need to do anything else in this file.
# The `main.py` script will automatically discover and load agents from this directory.