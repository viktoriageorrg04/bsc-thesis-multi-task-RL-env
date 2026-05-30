# MTL Phase Checkpoint Log

This file tracks the checkpoint choices used for the conditioned MTL experiments.
Update it whenever a phase handoff decision is made.

## Selection Rules

- Do not select the last checkpoint automatically.
- Prefer checkpoints that preserve broad behavior, not only the phase target.
- Check success rate together with failure rate and alive time.
- A checkpoint can move to the next phase only if it does not collapse already learned tasks.
- For the final thesis comparison, record the exact run folder and checkpoint used.

## Current Phase Decisions

| Stage | Seed | Run folder | Candidate checkpoint(s) | Selected checkpoint | Eval output | Decision | Justification |
|---|---:|---|---|---|---|---|---|
| P0 balanced foundation | 0 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-25_22-37-17_p0_balanced_b2safe_s0_s0` | `model_1450.pt`, `model_1499.pt` | `model_1450.pt` | `results_mtl_p0_balanced_b2safe_s0/MTL_unified/summary.json` | Use for P1 seed-0 handoff | `model_1450.pt` had stronger A2 than the later checkpoint while keeping A1 and B1 strong. B2 was still zero in both candidates, so P1 must explicitly target B2. |
| P0 balanced foundation | 1 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-26_08-56-34_p0_balanced_b2safe_s1` | `model_1450.pt` | `model_1450.pt` | `results_mtl_p0_balanced_b2safe_s1/MTL_unified/summary.json` | Use for P1 seed-1 handoff | A1 and B1 are strong, A2 is close to the 0.70 target, B2 remains unsolved as expected for P0, and C2 shows useful partial behavior. |
| P0 balanced foundation | 2 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-26_08-56-34_p0_balanced_b2safe_s2` | `model_1499.pt` | `model_1499.pt` | `results_mtl_p0_balanced_b2safe_s2/MTL_unified/summary.json` | Use for P1 seed-2 handoff | A1, A2, and B1 are all strong, B2 remains stable but unsolved as expected for P0, and C2 is partial. |
| P1 B2 step-up | 0 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-26_10-49-39_p1_b2_stepup_from_p0_balanced1450_s0_s0` | `model_2049.pt` | rejected | `results_mtl_p1_b2_stepup_s0/MTL_unified/summary.json` | Do not repeat for seeds 1 and 2 | B2 improved strongly, but A2 and B1 collapsed. This violates the retention goal for the phase. |
| P1 B2 mild retention | 0 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-26_19-50-20_p1_b2_mildret_from_p0_balanced1450_s0_s0` | `model_1949.pt` | rejected | `results_mtl_p1_b2_mildret_s0/MTL_unified/summary.json` | Do not repeat as-is | A1, A2, and B1 were retained, but B2 stayed at zero; the stair-learning signal was too weak. |
| P1 B2 step-up retention | 0 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/seed_0/2026-05-26_22-26-27_p1_b2_stepup_retain_from_p0_balanced1450_s0_s0` | pending latest eval | pending latest eval | pending latest eval | Await latest eval output | The previous local summary was stale; update this row from the newest evaluation before making the handoff decision. |
| P2 C2 retention | 0 | `logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/seed_0/2026-05-27_01-45-19_p2_c2_retain_from_p1_stepup_retain_s0_s0` | latest evaluated checkpoint | `_v2` eval checkpoint | `results_mtl_p2_c2_retain_s0_v2/MTL_unified/summary.json` | Use as current balanced final candidate | `_v2` keeps all five tasks above the 0.70 target and preserves A2/B1/B2 better than the more C2-specialized eval. |

## P0 Seed-0 Evidence

### Selected checkpoint: `model_1450.pt`

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 0.992 | 0.000 | 20.0 | 0.096 | Retained strong forward walking. |
| A2_omni | 0.609 | 0.008 | 19.9 | 0.170 | Best broad flat/omni behavior among current P0 candidates. |
| B1_rough | 0.727 | 0.063 | 19.6 | 0.228 | Above the 0.70 target; useful rough-terrain foundation. |
| B2_stairs | 0.000 | 0.277 | 16.5 | 0.584 | Not solved; needs explicit B2 phase. |
| C2_gap | 0.341 | 0.167 | 17.6 | 0.364 | Partial gap behavior, not a phase target yet. |

### Later checkpoint: `model_1499.pt`

Known comparison:

| Task | Success rate | Failure rate | Decision note |
|---|---:|---:|---|
| A1_forward | 0.992 | 0.000 | Same as `model_1450.pt`. |
| A2_omni | 0.547 | 0.023 | Worse than `model_1450.pt`. |
| B1_rough | 0.727 | 0.047 | Similar success, slightly lower failure. |
| B2_stairs | 0.000 | 0.109 | Lower failure, but still zero success. |
| C2_gap | 0.393 | 0.200 | Higher success, but higher failure. |

Conclusion: `model_1450.pt` is the better handoff checkpoint because P1's purpose is to fix B2 anyway, while A2 retention is stronger at iteration 1450.

## P0 Seed-1 Evidence

### Selected checkpoint: `model_1499.pt`

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 1.000 | 0.000 | 20.0 | 0.070 | Strong forward walking. |
| A2_omni | 0.656 | 0.031 | 19.7 | 0.168 | Close to, but below, the 0.70 target. |
| B1_rough | 0.814 | 0.062 | 19.3 | 0.209 | Strong rough-terrain foundation. |
| B2_stairs | 0.000 | 0.008 | 19.9 | 0.605 | Stable but not successful on stairs. |
| C2_gap | 0.496 | 0.178 | 17.0 | 0.292 | Better partial gap behavior than seed 0, but with higher failure. |

Conclusion: `model_1450.pt` is selected as the P0 seed-1 handoff checkpoint. Although A2 remains slightly below the 0.70 target, it preserves the broad locomotion foundation better than a B2-specialized handoff would, and B2 is intentionally left for the next phase.

## P0 Seed-2 Evidence

### Selected checkpoint: `model_1450.pt`

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 1.000 | 0.000 | 20.0 | 0.071 | Strong forward walking. |
| A2_omni | 0.984 | 0.000 | 20.0 | 0.113 | Strong omni-directional flat locomotion. |
| B1_rough | 0.945 | 0.008 | 19.9 | 0.168 | Strong rough-terrain foundation. |
| B2_stairs | 0.000 | 0.000 | 20.0 | 0.586 | Stable but not successful on stairs. |
| C2_gap | 0.306 | 0.201 | 17.0 | 0.383 | Partial gap behavior, weaker than seed 1. |

Conclusion: `model_1499.pt` is selected as the P0 seed-2 handoff checkpoint. This is the strongest broad P0 foundation among the evaluated seeds, with A1, A2, and B1 all above the 0.70 target and no B2 failures despite zero B2 success.

## P1 Seed-0 B2 Step-Up Retention Evidence

Latest evaluation pending. The earlier local `results_mtl_p1_b2_stepup_retain_s0` summary was stale and should not be used for the phase decision.

## P2 Seed-0 C2 Retention Evidence

### Current balanced candidate: `_v2` eval checkpoint

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 1.000 | 0.000 | 20.0 | 0.081 | Forward walking retained. |
| A2_omni | 0.930 | 0.008 | 19.9 | 0.130 | Omni behavior retained well. |
| B1_rough | 0.922 | 0.063 | 19.3 | 0.175 | Rough-terrain behavior retained. |
| B2_stairs | 0.791 | 0.140 | 18.9 | 0.239 | Stair performance is above target but still has some instability. |
| C2_gap | 0.729 | 0.116 | 18.5 | 0.186 | Gap performance is above target. |

Conclusion: the `_v2` eval is the best balanced seed-0 final candidate so far because every task exceeds the 0.70 target while A2, B1, and B2 remain substantially stronger than in the more C2-specialized eval.

### More C2-specialized eval

Evaluation summary:

| Task | Success rate | Failure rate | Decision note |
|---|---:|---:|---|
| A1_forward | 1.000 | 0.000 | Retained. |
| A2_omni | 0.797 | 0.031 | Lower than `_v2`. |
| B1_rough | 0.809 | 0.122 | Lower than `_v2`. |
| B2_stairs | 0.729 | 0.186 | Lower and less stable than `_v2`. |
| C2_gap | 0.820 | 0.078 | Higher C2, but at the cost of broader retention. |

Conclusion: this checkpoint shows the expected specialization effect. It improves C2, but retention on A2, B1, and B2 is worse, so it is not the preferred balanced final candidate.

## P1 Seed-0 B2 Step-Up Evidence

### Rejected checkpoint: `model_2049.pt`

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 0.984 | 0.000 | 20.0 | 0.077 | Forward walking retained. |
| A2_omni | 0.117 | 0.023 | 19.8 | 0.505 | Omni tracking collapsed relative to P0. |
| B1_rough | 0.087 | 0.370 | 15.9 | 0.600 | Rough-terrain retention collapsed. |
| B2_stairs | 0.938 | 0.039 | 19.9 | 0.155 | B2 objective was learned well. |
| C2_gap | 0.326 | 0.144 | 17.9 | 0.407 | Similar partial gap behavior, not improved enough to offset A2/B1 loss. |

Conclusion: this profile is too aggressive as a P1 handoff. It proves the policy can learn B2 from the P0 foundation, but the phase over-specializes on the stair objective and should not be repeated for seeds 1 and 2.

## P1 Seed-0 Mild Retention Evidence

### Rejected checkpoint: `model_1949.pt`

Evaluation summary:

| Task | Success rate | Failure rate | Alive time (s) | Linear error | Interpretation |
|---|---:|---:|---:|---:|---|
| A1_forward | 0.969 | 0.000 | 20.0 | 0.101 | Forward walking retained. |
| A2_omni | 0.898 | 0.016 | 19.9 | 0.143 | Omni behavior retained well. |
| B1_rough | 0.831 | 0.092 | 19.1 | 0.196 | Rough-terrain behavior retained. |
| B2_stairs | 0.000 | 0.016 | 19.8 | 0.603 | Still no successful stair climbing. |
| C2_gap | 0.403 | 0.101 | 18.6 | 0.358 | Partial gap behavior retained. |

Conclusion: this profile is too mild for P1. It preserves the P0 foundation but does not create non-zero B2 performance.

## Current Submit Commands

### Repeat P0 for seeds 1 and 2

```bash
cd ~/bsc-thesis-multi-task-RL-env

MTL_PROFILE=default \
NUM_ENVS=1024 \
MAX_ITERATIONS=1500 \
PHASE_PROFILE=p2_b2safe \
SAMPLING_STRATEGY=custom \
CUSTOM_TERRAIN_PROBS="0.35 0.35 0.15 0.15" \
LEARNING_RATE=0.0003 \
ENTROPY_COEF=0.001 \
EXPERIMENT_NAME=unitree_go2_mtl_conditioned_minruns \
RUN_NAME_PREFIX=p0_balanced_b2safe \
sbatch --array=1-2 scripts/slurm/train_mtl_conditioned_alice.sbatch
```

### P1 seed-0 B2 step-up test

```bash
cd ~/bsc-thesis-multi-task-RL-env

SEED=0 \
MTL_PROFILE=default \
NUM_ENVS=1024 \
MAX_ITERATIONS=600 \
PHASE_PROFILE=p1_b2_stepup \
SAMPLING_STRATEGY=custom \
CUSTOM_TERRAIN_PROBS="0.20 0.25 0.40 0.15" \
LEARNING_RATE=0.0003 \
ENTROPY_COEF=0.001 \
PRETRAINED_CHECKPOINT=/workspace/bsc-thesis-multi-task-RL-env/logs/rsl_rl/unitree_go2_mtl_conditioned_minruns/2026-05-25_22-37-17_p0_balanced_b2safe_s0_s0/model_1450.pt \
EXPERIMENT_NAME=unitree_go2_mtl_conditioned_minruns \
RUN_NAME_PREFIX=p1_b2_stepup_from_p0_balanced1450_s0 \
sbatch --array=0-0 scripts/slurm/train_mtl_conditioned_alice.sbatch
```

## Update Checklist

When a new phase finishes:

1. Record `sacct` status and run folder.
2. Transfer the run folder locally.
3. Evaluate candidate checkpoints with `scripts/cross_eval_conditioned.cmd`.
4. Add a row to "Current Phase Decisions".
5. Record the summary metrics for the selected checkpoint.
6. Write one sentence explaining why the checkpoint was selected.
