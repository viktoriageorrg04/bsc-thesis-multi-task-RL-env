"""Environments: task families, rewards, observations, actions, success logic."""

from .registry import register_tasks

register_tasks()  # call the function to register all envs when this module is imported