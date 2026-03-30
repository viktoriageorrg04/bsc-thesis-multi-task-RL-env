"""Gymnasium task registration for all benchmark envs."""

from .families import agility_terrain, flat_velocity, multi_task, rough_velocity

def register_tasks() -> None:
    # import side-effects register gym IDs in each fam package
    _ = (flat_velocity, rough_velocity, agility_terrain, multi_task)

