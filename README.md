# Multi-Task RL Locomotion Benchmark

BSc Thesis — Data Science and AI  
**Author:** Viktoria Georgieva

A multi-task reinforcement-learning benchmark for locomotion, built on
[Isaac Lab](https://isaac-sim.github.io/IsaacLab/).  The project defines a
reusable task interface, a success-based evaluation protocol, and a small
suite of locomotion task families spanning flat, rough, and custom terrains.

## Project layout

```
├── envs/                   # Phase 1 + 2 — task & env definitions
│   ├── families/           # one sub-package per task family
│   │   ├── flat_velocity/  # Fam A  (Isaac Lab built-in wrappers)
│   │   ├── rough_velocity/ # Fam B  (Isaac Lab built-in wrappers)
│   │   └── agility_terrain/ # Fam C  (agility terrains — stepping stones, gaps)
│   ├── rewards/            # reusable reward components / terms
│   ├── observations/       # shared obs-space definitions
│   ├── actions/            # shared action conventions
│   ├── success.py          # unified success / termination logic
│   └── registry.py         # Gymnasium task registration
│
├── benchmark/              # Phase 2 + 3 — benchmark layer
│   ├── interface.py        # std Gymnasium wrapper exposed to algos
│   ├── curriculum.py       # curriculum / task-staging logic
│   ├── task_sampler.py     # multi-task sampling (Task ID vs No Task ID)
│   └── metrics.py          # success rate, learning speed, transfer perf
│
├── configs/                # YAML / Python training & eval configs
│   ├── tasks/              # per-task env configs (terrain params, etc.)
│   ├── training/           # algo hyper-params (PPO, optional SAC/TD3)
│   └── eval/               # eval-protocol configs (scaling, held-out, …)
│
├── scripts/                # Phase 3 — runnable entry points
│   ├── train.py            # single / multi-task training driver
│   ├── evaluate.py         # run eval protocol & collect metrics
│   └── visualize.py        # render / record episodes
│
├── analysis/               # Phase 4 — post-hoc analysis & plots
│   ├── compare.py          # single-task vs multi-task comparisons
│   ├── transfer.py         # cross-family transfer analysis
│   ├── reward_sensitivity.py  # reward-scaling sensitivity study
│   └── plots/              # saved figures (gitignored except README)
│
├── tests/                  # unit & integration tests
│   ├── test_envs.py
│   ├── test_rewards.py
│   └── test_benchmark.py
│
├── docs/                   # extra documentation / notes
│   └── eval_protocol.md    # eval-protocol spec (for thesis appendix)
│
├── .gitignore
├── pyproject.toml
└── README.md
```

## Quick start

```bash
# 1. clone & enter
git clone <repo-url> && cd bsc-thesis-multi-task-RL-env

# 2. install (editable, inside an Isaac Lab conda/venv)
pip install -e .

# 3. train (example)
python scripts/train.py --config configs/training/ppo_flat.yaml

# 4. evaluate
python scripts/evaluate.py --config configs/eval/scaling_test.yaml
```

## Phases overview

| Phase | Focus | Key deliverable |
|-------|-------|-----------------|
| 1 | Task space (3 families x 2-3 tasks) | `envs/families/` |
| 2 | Std interface (obs, act, rew, success) | `envs/` + `benchmark/` |
| 3 | Eval protocol (scaling, Task ID, success) | `scripts/` + `configs/eval/` |
| 4 | Analysis (single vs multi, transfer, reward) | `analysis/` |