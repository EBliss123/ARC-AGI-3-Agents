"""
Purpose: The Central Orchestrator.
Responsibilities:
1. Initializes the environment, memory, and engines.
2. Runs the main execution loop (Predict -> Act -> Observe -> Verify).
3. Passes tensors between modules. Contains NO math or logic of its own.
"""