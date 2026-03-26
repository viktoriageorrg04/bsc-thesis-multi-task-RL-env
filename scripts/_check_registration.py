"""verify all MTL gym IDs register"""
import sys
sys.path.insert(0, r"C:\Users\pavel\bsc-thesis-multi-task-RL-env")

import envs
import gymnasium as gym

mtl = sorted(k for k in gym.registry if k.startswith("MTL-"))
print(f"Registered {len(mtl)} MTL envs:")
for name in mtl:
    print(f"  {name}")
