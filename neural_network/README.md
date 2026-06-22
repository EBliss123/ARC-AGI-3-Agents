# ARC-AGI-3 Hierarchical Meta-Learning Engine

## Project Context
This repository contains a machine learning architecture designed for the **ARC-AGI-3 (Abstraction and Reasoning Corpus) Kaggle Competition**. The goal of ARC is to measure general fluid intelligence by forcing AI agents to solve novel spatial reasoning puzzles using only a few demonstration examples. 

Standard reinforcement learning and computer vision techniques fail at ARC because they attempt to memorize specific puzzle layouts. To solve this, this project implements a **Hierarchical Meta-Learning Architecture** that completely separates the discovery of universal laws (Long-Term Planner) from the localized execution of pure physics logic (Short-Term Planner).

---

## Project Structure & File Manifest
The engine is highly modularized to separate the Kaggle emulator interactions from the PyTorch mathematics.

* **`main.py` (The Execution Hub):** The entry point. It initializes the ARC emulator, manages the live game loop, captures state-action-reward trajectories, and triggers the real-time learning updates after every move.
* **`networks.py` (The Brains):** Contains the PyTorch neural network classes. This houses the 4097-Node Self-Attention Transformer used by the Short-Term Planner, including the exact logic for the Change Mask, Color Target, and Reactive Policy heads.
* **`meta_loop.py` (The Learning Calculus):** Houses the PyTorch optimization algorithms. It contains the decoupled physics and policy optimizers, the fast-adaptation formulas for real-time learning, and the Gradient-Through-A-Gradient logic used to update the universal basecamp.
* **`data_utils.py` (The Translators):** Handles the objective translation of the game state. It converts the raw Kaggle grid into the normalized `64x64x18` mathematical tensors and structures the 10-channel Global Action Node.

---

## 1. The Long-Term Planner (The Scientist / Basecamp Meta-Learning)
The Long-Term Planner (LTP) acts as an accelerator, not a physics engine. 
* **Curriculum Focus:** It trains exclusively on solving Level 1 across the 25 games, preventing noisy gradients from complex, multi-step levels.
* **The Basecamp:** Once the Short-Term Planners successfully solve their respective Level 1s, the LTP averages those successful weights together. This calculates an "Universal Basecamp"—a highly optimized, static starting point. 
* **The Goal:** The LTP does not know the specific physics of any game. Instead, its weights naturally encode foundational ARC laws (boundaries, movement, color-matching). When dropped into a new, unseen Level 2+, the agent spawns at this basecamp, allowing it to adapt to complex mechanics in a fraction of the time.

## 2. The Short-Term Planner (The Tactician / Objective Physics Engine)
The Short-Term Planner (STP) lives exclusively inside a single game to discover its unique physics. To maintain universal applicability, the STP operates entirely on pure, decentralized logic without high-level "actor" or "object" labels.

### Objective Vision (The 4097-Node Transformer)
Standard Convolutional Neural Networks (CNNs) rely on sliding windows that destroy absolute coordinate logic. To preserve exact spatial reality, the STP completely bypasses CNNs in favor of a **Self-Attention** architecture. 

The input is flattened into exactly 4,097 independent nodes:
* **Nodes 0 to 4095 (The Grid):** Every pixel is an independent node with 18 channels (16 one-hot colors, Normalized X, Normalized Y). Every pixel mathematically knows exactly what it is and where it is in space.
* **Node 4096 (The Global Action Node):** A single super-node containing a 10-channel vector (8 one-hot discrete buttons, Normalized X, Normalized Y).

Through Self-Attention, the 4,096 pixel nodes send mathematical queries to each other and to the Action Node. This natively allows the network to calculate direct conditional events across the entire board instantly, recognizing exact coordinate triggers without relying on brittle localized windows.

### The Pure Logic Calculus
Every single turn, the STP evaluates the 4097 nodes. It processes physics in two stages:
1. **The Change Mask:** A binary prediction layer. Every pixel independently evaluates if its conditions have been met based on its coordinates and the Action Node. It outputs `1` if it predicts it will change on the next frame, and `0` if it will stay the same.
2. **The Color Target:** For pixels that flagged a `1`, a secondary layer evaluates the conditions again to predict the new color channel.

Because the math is applied pixel-by-pixel, the Categorical Cross-Entropy Loss function surgically corrects the network if it mispredicts a specific pixel's behavior, creating a highly accurate, objective physics simulator.

## 3. Action Planning & Decision Engine (Gut Instinct)
The Decision Engine dictates *what* button to press, acting completely independently of the Physics Engine's predictions.

It uses a **Reactive Policy Layer** that processes the board state and outputs a probability distribution for all valid moves. It trains via **REINFORCE (Curiosity Rewards)**:
* **Reward (`+1.0`):** If the chosen action results in a novel physical change to the board, it receives a positive reward, boosting the mathematical confidence of that specific decision path.
* **Penalty (`-1.0`):** If the chosen action results in `NO_CHANGE` (a wasted click), it receives a heavy penalty, mathematically forcing the network to become "allergic" to useless interactions.

This decoupling ensures the agent learns optimal exploratory behavior without being paralyzed by temporary inaccuracies in its own physics predictions.

---

## Execution
Run the engine from the `neural_network` directory using the virtual environment:
```bash
venv\Scripts\activate
python main.py