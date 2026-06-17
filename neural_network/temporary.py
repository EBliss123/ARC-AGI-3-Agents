import arc_agi

# Initialize and load vc33
arc = arc_agi.Arcade()
env = arc.make("vc33")

# 1. Inspect the Action Space
print("--- Action Space ---")
print(f"Type: {type(env.action_space)}")
print(f"Attributes: {dir(env.action_space)}")

# 2. Inspect the Observation (Kaggle sometimes attaches valid actions here)
obs = env.reset()
print("\n--- Observation Attributes ---")
print([attr for attr in dir(obs) if not attr.startswith('_')])