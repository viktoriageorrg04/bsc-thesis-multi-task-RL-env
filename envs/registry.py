"""Gymnasium task registration for all benchmark envs."""

from .families import custom_terrain, flat_velocity, rough_velocity

def register_tasks() -> None:
    # import side-effects register gym IDs in each fam package
    _ = (flat_velocity, rough_velocity, custom_terrain)

