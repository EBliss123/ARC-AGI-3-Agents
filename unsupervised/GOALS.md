# ARC-AGI-3 Unsupervised Agent Roadmap

### Milestone 1: Raw JSON Ingestion (The Observer)
**Goal:** Read the actual `ls20` environment data seamlessly. 
**Details:** The ARC-AGI-3 environment returns states where a single action can trigger multiple frames if an animation cascade occurs. The `arc_interface.py` and `observer.py` must catch this 1-to-N frame sequence and successfully output our unified `[x, y, old_color, new_color]` PyTorch tensors for *each* frame transition without crashing.

### Milestone 2: Unsupervised Bloat (The First Pass)
**Goal:** The Recursive Compressor groups objects naturally and writes functional, brute-force physics rules.
**Details:** We do not care about elegance yet. The engine uses its Tier 1 and Tier 2 math to group pixels that share a fate. The resulting JSON rules will be massive and highly bloated (e.g., tracking irrelevant background pixels), but they must perfectly predict the next frame.

### Milestone 3: Historical Compression (MDL Activation)
**Goal:** The engine rewrites its own memory to be more efficient without breaking past predictions.
**Details:** As the agent sees more of the game, bloated rules will inevitably fail. When a rule breaks, the Minimum Description Length (MDL) protocol activates. The engine tests new hypotheses against *all historical frames*, redefining its object sets and stripping away variables that aren't strictly necessary.

### Milestone 4: Human-Level Abstraction
**Goal:** The Tiered Search bottoms out into hyper-elegant, minimal JSON logic.
**Details:** The engine successfully crushes the bloated rules down. It completely shifts from absolute existence (e.g., "Pixel at 10,10") to relative invariants (e.g., "Set A is +1X from Color B"), perfectly mirroring human-level physics comprehension. 

### Milestone 5: Win Conditions & Neural Net Policy (ON HOLD)
*Disclaimer: This phase is strictly locked until Milestones 1-4 are flawless.*
**Details:** Once the agent can perfectly map the physics of any game unsupervised, we will activate the Win Intersector to find the commonalities between the physics and the win conditions. Only then will we train a neural network policy to know which actions (`ACTION1` through `ACTION7`) are best to try to achieve that extracted win state. Most importantly, this is where it is going to compare win conditions with physics data in the neural network to decipher rules for choosing best actions.