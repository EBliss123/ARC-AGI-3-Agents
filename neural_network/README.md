# ARC-AGI-3 Meta-Learning Engine

## Architecture Overview
This project implements a test-time training (meta-learning) architecture to solve spatial reasoning puzzles in the ARC-AGI-3 Kaggle environment. 

Instead of training a monolithic reinforcement learning agent, this engine separates "planning" from "physics prediction." It relies on a global baseline model that spawns temporary clones to rapidly adapt to the physics of individual ARC games within a 30-step limit.

## Core Loop
The system uses a **Gradient-Through-A-Gradient** approach:
1. **The Inner Loop (Fast Adaptation):** For a specific game, a temporary clone of the network is spawned. As it interacts with the emulator, it records (Input Grid, Action Taken, Actual Next Grid). The clone uses an aggressive SGD optimizer (`lr=0.05`) to update its Action Network (fast weights) to minimize the surprise between its predicted frame and the actual frame.
2. **The Outer Loop (Generalization):** After adaptation, the clone makes a final move on an unseen state. The loss from this final prediction is passed backward all the way through the adaptation steps to update the Global Model using Adam (`lr=0.0005`). This teaches the global model how to initialize better starting weights for future clones.

## Neural Networks (`networks.py`)
The architecture is split into three interconnected modules:

### 1. Planner Network
* **Role:** Decides *what* button to press and *where* to click.
* **Input:** The 64x64 grid (one-hot encoded across 16 color channels).
* **Output:** 4,104 probabilities (8 discrete buttons + 4,096 spatial coordinates).
* **Guardrails:** Uses a dynamic `action_mask` based on the emulator's `available_actions` to strictly force the probabilities of invalid moves to 0%.

### 2. Action Network (Fast Weights)
* **Role:** Translates the chosen action into physical modifiers. 
* **Input:** A unified 136-node action vector (Discrete ID + X + Y).
* **Output:** 2,048 FiLM (Feature-wise Linear Modulation) parameters (gamma multipliers and beta shifts).
* **Note:** This is the *only* network that is updated during the Inner Loop. It learns the specific physical rules of the current game on the fly.

### 3. Predictor Network (Slow Weights)
* **Role:** Predicts the exact next 64x64 frame.
* **Mechanism:** It processes the current grid and dynamically injects the FiLM parameters from the Action Network at each hidden layer, effectively altering its own internal logic based on the action taken.
* **Residual Connection Bias:** Because the vast majority of pixels in ARC games do not change per turn, the Predictor takes the original input grid, multiplies it by a heavy bias (`* 10.0`), and adds it to the final output. This mathematically forces the network to default to "No Change" and focus exclusively on predicting the deltas.

## Data & Logic Heuristics (`data_utils.py` & `main.py`)

### Dynamic Guardrails
The Kaggle Arcade emulator is strictly enforced as the ground truth. At the start of every turn, `obs.available_actions` is parsed to disable any invalid discrete buttons. The `RESET` button is hard-disabled to prevent the agent from throwing away its 30-step learning trajectory.

### Contiguous Spatial Clustering
If the model decides to click the grid (`ACTION6`), it cannot choose completely at random. `scipy.ndimage.label` scans the raw grid for 4-way contiguous color blobs. The `action_mask` disables all 4,096 spatial nodes *except* for one representative coordinate per valid cluster. This drastically reduces the action space and forces the agent to interact with actual objects.

### Verification Tracking
The inner loop checks if `np.array_equal(raw_grid, next_raw_grid)` after every move. Actions that result in no visual change (e.g., tool selections) are flagged in the logs with `(NO_CHANGE)`.

## Execution
Run the engine from the `neural_network` directory using the virtual environment:
```bash
venv\Scripts\activate
python main.py