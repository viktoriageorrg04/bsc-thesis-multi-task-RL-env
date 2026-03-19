"""Tests for benchmark interface, task sampling, and metrics.

All tests here exercise the deterministic success/termination logic
(envs.success) and the EpisodeStatsTracker using synthetic tensors —
no Isaac Sim runtime needed.

Boundary conditions tested:
  - near-threshold (just below / just above eps_lin and eps_ang)
  - timeout path (truncated=True, terminated=False)
  - fall / failure path (terminated=True mid-episode)
  - stall (zero tracking, always within threshold -> success)
"""

import pytest
import torch

from envs.success import (
    DEFAULT_SUCCESS_CONFIG,
    EpisodeStatsTracker,
    SuccessConfig,
    compute_step_success_from_errors,
)

from benchmark.task_sampler import (
    ALL_TRAIN_TASK_IDS,
    TASK_ID_OFF,
    TASK_ID_ON,
    TaskIDConfig,
    TaskSampler,
)

# defaults: eps_lin=0.25, eps_ang=0.50, min_success_ratio=0.80
CFG = DEFAULT_SUCCESS_CONFIG
NUM_ENVS = 4
STEP_DT = 0.02


# helpers

def _ones(n: int = NUM_ENVS) -> torch.Tensor:
    return torch.ones(n)


def _zeros(n: int = NUM_ENVS) -> torch.Tensor:
    return torch.zeros(n)


def _bools(values: list[bool]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool)


def _make_tracker(num_envs: int = NUM_ENVS) -> EpisodeStatsTracker:
    return EpisodeStatsTracker(num_envs=num_envs, step_dt=STEP_DT, cfg=CFG, device="cpu")


def _update_kwargs(
    *,
    step_success: torch.Tensor,
    lin_err: torch.Tensor,
    ang_err: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    task_id: int = 0,
    family_id: int = 0,
) -> dict:
    """Build keyword args for EpisodeStatsTracker.update()."""
    return dict(
        step_success=step_success,
        lin_err_xy=lin_err,
        ang_err_z=ang_err,
        terminated=terminated,
        truncated=truncated,
        task_id=task_id,
        family_id=family_id,
    )


# compute_step_success_from_errors

class TestStepSuccess:
    """Unit tests for the per-step success predicate."""

    def test_all_within_threshold(self):
        """All envs alive, errors well below thresholds → all True."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((NUM_ENVS,), 0.10),
            ang_err_z=torch.full((NUM_ENVS,), 0.20),
            terminated=_zeros(),
            cfg=CFG,
        )
        assert result.all()

    def test_all_above_threshold(self):
        """All envs alive, errors above both thresholds → all False."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((NUM_ENVS,), 0.50),
            ang_err_z=torch.full((NUM_ENVS,), 1.00),
            terminated=_zeros(),
            cfg=CFG,
        )
        assert not result.any()

    def test_terminated_always_fails(self):
        """Even with zero errors, terminated envs are not successful."""
        result = compute_step_success_from_errors(
            lin_err_xy=_zeros(),
            ang_err_z=_zeros(),
            terminated=_ones(),
            cfg=CFG,
        )
        assert not result.any()

    # near-threshold boundary tests

    def test_lin_err_exactly_at_threshold(self):
        """lin_err == eps_lin  →  still success (<=)."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((1,), CFG.eps_lin),
            ang_err_z=torch.zeros(1),
            terminated=torch.zeros(1),
        )
        assert result.item() is True

    def test_lin_err_just_above_threshold(self):
        """lin_err slightly above eps_lin  →  fail."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((1,), CFG.eps_lin + 1e-6),
            ang_err_z=torch.zeros(1),
            terminated=torch.zeros(1),
        )
        assert result.item() is False

    def test_ang_err_exactly_at_threshold(self):
        """ang_err == eps_ang  →  still success (<=)."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.zeros(1),
            ang_err_z=torch.full((1,), CFG.eps_ang),
            terminated=torch.zeros(1),
        )
        assert result.item() is True

    def test_ang_err_just_above_threshold(self):
        """ang_err slightly above eps_ang  →  fail."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.zeros(1),
            ang_err_z=torch.full((1,), CFG.eps_ang + 1e-6),
            terminated=torch.zeros(1),
        )
        assert result.item() is False

    def test_lin_ok_ang_over(self):
        """lin within threshold but ang over  →  fail."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((1,), 0.10),
            ang_err_z=torch.full((1,), 0.80),
            terminated=torch.zeros(1),
        )
        assert result.item() is False

    def test_ang_ok_lin_over(self):
        """ang within threshold but lin over  →  fail."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.full((1,), 0.40),
            ang_err_z=torch.full((1,), 0.10),
            terminated=torch.zeros(1),
        )
        assert result.item() is False

    def test_mixed_envs(self):
        """Different envs hit different boundary cases in one batch."""
        result = compute_step_success_from_errors(
            lin_err_xy=torch.tensor([0.10, 0.30, 0.10, 0.00]),
            ang_err_z=torch.tensor([0.10, 0.10, 0.60, 0.00]),
            terminated=torch.tensor([0.0, 0.0, 0.0, 1.0]),
            cfg=CFG,
        )
        # env0: both ok, alive -> True
        # env1: lin over -> False
        # env2: ang over -> False
        # env3: terminated -> False
        assert result.tolist() == [True, False, False, False]


# EpisodeStatsTracker

class TestEpisodeStatsTracker:
    """Integration tests for episode-level success aggregation."""

    # timeout path

    def test_timeout_all_success_steps(self):
        """100% successful steps + timeout -> episode_success=True."""
        tracker = _make_tracker(num_envs=1)
        # simulate 10 perfect steps, then timeout on step 10
        for i in range(10):
            is_last = i == 9
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
                task_id=1,
                family_id=0,
            ))
        assert len(rows) == 1
        ep = rows[0]
        assert ep["episode_success"] is True
        assert ep["success_step_ratio"] == pytest.approx(1.0)
        assert ep["termination_reason"] == "time_out"
        assert ep["alive_time_s"] == pytest.approx(10 * STEP_DT)
        assert ep["task_id"] == 1
        assert ep["family_id"] == 0

    def test_timeout_below_min_ratio(self):
        """<80% success steps + timeout -> episode_success=False."""
        tracker = _make_tracker(num_envs=1)
        # 10 steps; first 7 bad, last 3 good -> 30% ratio
        for i in range(10):
            is_last = i == 9
            good = i >= 7
            rows = tracker.update(**_update_kwargs(
                step_success=torch.tensor([float(good)]),
                lin_err=torch.full((1,), 0.10 if good else 0.40),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
            ))
        assert len(rows) == 1
        assert rows[0]["episode_success"] is False
        assert rows[0]["success_step_ratio"] == pytest.approx(0.3)
        assert rows[0]["termination_reason"] == "time_out"

    def test_timeout_exactly_at_min_ratio(self):
        """Exactly 80% success + timeout -> episode_success=True."""
        tracker = _make_tracker(num_envs=1)
        # 10 steps; first 8 good, last 2 bad
        for i in range(10):
            is_last = i == 9
            good = i < 8
            rows = tracker.update(**_update_kwargs(
                step_success=torch.tensor([float(good)]),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
            ))
        assert len(rows) == 1
        assert rows[0]["episode_success"] is True
        assert rows[0]["success_step_ratio"] == pytest.approx(0.8)

    def test_timeout_just_below_min_ratio(self):
        """79% success + timeout -> episode_success=False (boundary)."""
        tracker = _make_tracker(num_envs=1)
        n_steps = 100
        n_good = 79  # 79%
        for i in range(n_steps):
            is_last = i == n_steps - 1
            good = i < n_good
            rows = tracker.update(**_update_kwargs(
                step_success=torch.tensor([float(good)]),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
            ))
        assert len(rows) == 1
        assert rows[0]["episode_success"] is False
        assert rows[0]["success_step_ratio"] == pytest.approx(0.79)

    # fall / failure path

    def test_failure_overrides_success(self):
        """Even 100% success steps: if terminated (fall) -> episode fails."""
        tracker = _make_tracker(num_envs=1)
        for i in range(5):
            is_last = i == 4
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.tensor([float(is_last)]),
                truncated=torch.zeros(1),
            ))
        assert len(rows) == 1
        assert rows[0]["episode_success"] is False
        assert rows[0]["termination_reason"] == "failure"
        assert rows[0]["success_step_ratio"] == pytest.approx(1.0)

    def test_early_failure(self):
        """Terminated on 1st step -> failure, 1 step, 0% success."""
        tracker = _make_tracker(num_envs=1)
        rows = tracker.update(**_update_kwargs(
            step_success=torch.zeros(1),
            lin_err=torch.full((1,), 0.50),
            ang_err=torch.full((1,), 0.10),
            terminated=torch.ones(1),
            truncated=torch.zeros(1),
        ))
        assert len(rows) == 1
        ep = rows[0]
        assert ep["episode_success"] is False
        assert ep["termination_reason"] == "failure"
        assert ep["success_step_ratio"] == pytest.approx(0.0)
        assert ep["alive_time_s"] == pytest.approx(STEP_DT)

    def test_failure_mid_episode(self):
        """Fall on step 3 of 10 -> failure, only 3 steps counted."""
        tracker = _make_tracker(num_envs=1)
        for i in range(3):
            is_last = i == 2
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.tensor([float(is_last)]),
                truncated=torch.zeros(1),
            ))
        assert len(rows) == 1
        assert rows[0]["episode_success"] is False
        assert rows[0]["termination_reason"] == "failure"
        assert rows[0]["alive_time_s"] == pytest.approx(3 * STEP_DT)

    # stall path

    def test_stall_perfect_tracking(self):
        """Zero errors every step + timeout -> episode_success=True."""
        tracker = _make_tracker(num_envs=1)
        for i in range(20):
            is_last = i == 19
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.zeros(1),
                ang_err=torch.zeros(1),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
            ))
        assert len(rows) == 1
        ep = rows[0]
        assert ep["episode_success"] is True
        assert ep["mean_lin_vel_error_xy"] == pytest.approx(0.0)
        assert ep["mean_ang_vel_error_z"] == pytest.approx(0.0)

    # buffer reset

    def test_buffers_reset_after_done(self):
        """After an episode completes, tracker resets for new episode."""
        tracker = _make_tracker(num_envs=1)
        # first episode: 5 steps → timeout
        for i in range(5):
            tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.full((1,), 0.10),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(i == 4)]),
            ))
        # after reset, counters should be zero
        assert tracker.total_steps[0].item() == 0.0
        assert tracker.success_steps[0].item() == 0.0
        assert tracker.lin_err_sum[0].item() == 0.0
        assert tracker.ang_err_sum[0].item() == 0.0
        assert tracker.had_failure[0].item() is False

        # second episode: 3 steps -> failure
        for i in range(3):
            rows = tracker.update(**_update_kwargs(
                step_success=torch.zeros(1),
                lin_err=torch.full((1,), 0.50),
                ang_err=torch.full((1,), 0.10),
                terminated=torch.tensor([float(i == 2)]),
                truncated=torch.zeros(1),
            ))
        assert len(rows) == 1
        assert rows[0]["alive_time_s"] == pytest.approx(3 * STEP_DT)

    # multi-env

    def test_multi_env_independent(self):
        """Env 0 times out while env 1 continues -> only env 0 emits."""
        tracker = _make_tracker(num_envs=2)
        # 5 steps for both; env 0 times out on step 5, env 1 continues
        for i in range(5):
            is_last = i == 4
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(2),
                lin_err=torch.full((2,), 0.10),
                ang_err=torch.full((2,), 0.10),
                terminated=torch.zeros(2),
                truncated=torch.tensor([float(is_last), 0.0]),
                task_id=3,
                family_id=2,
            ))
        # only env 0 emits
        assert len(rows) == 1
        assert rows[0]["episode_success"] is True
        assert rows[0]["task_id"] == 3
        assert rows[0]["family_id"] == 2
        # env 1 still accumulating
        assert tracker.total_steps[1].item() == 5.0

    def test_multi_env_simultaneous_done(self):
        """Both envs done on same step → 2 rows emitted."""
        tracker = _make_tracker(num_envs=2)
        for i in range(5):
            is_last = i == 4
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(2),
                lin_err=torch.full((2,), 0.10),
                ang_err=torch.full((2,), 0.10),
                terminated=torch.tensor([0.0, float(is_last)]),
                truncated=torch.tensor([float(is_last), 0.0]),
            ))
        assert len(rows) == 2
        # env 0: timeout -> success
        assert rows[0]["termination_reason"] == "time_out"
        assert rows[0]["episode_success"] is True
        # env 1: failure -> not success
        assert rows[1]["termination_reason"] == "failure"
        assert rows[1]["episode_success"] is False

    # mean error tracking

    def test_mean_errors_computed_correctly(self):
        """Mean lin/ang errors averaged over all steps in the episode."""
        tracker = _make_tracker(num_envs=1)
        lin_errs = [0.10, 0.20, 0.30]
        ang_errs = [0.05, 0.15, 0.25]
        for i, (le, ae) in enumerate(zip(lin_errs, ang_errs)):
            is_last = i == 2
            rows = tracker.update(**_update_kwargs(
                step_success=torch.ones(1),
                lin_err=torch.tensor([le]),
                ang_err=torch.tensor([ae]),
                terminated=torch.zeros(1),
                truncated=torch.tensor([float(is_last)]),
            ))
        assert len(rows) == 1
        assert rows[0]["mean_lin_vel_error_xy"] == pytest.approx(0.20, abs=1e-6)
        assert rows[0]["mean_ang_vel_error_z"] == pytest.approx(0.15, abs=1e-6)


# SuccessConfig

class TestSuccessConfig:
    """Verify SuccessConfig defaults and custom thresholds."""

    def test_default_thresholds(self):
        assert CFG.eps_lin == 0.25
        assert CFG.eps_ang == 0.50
        assert CFG.min_success_ratio == 0.80

    def test_custom_config_propagates(self):
        custom = SuccessConfig(eps_lin=0.10, eps_ang=0.20, min_success_ratio=0.90)
        # just below custom thresholds -> success
        result = compute_step_success_from_errors(
            lin_err_xy=torch.tensor([0.09]),
            ang_err_z=torch.tensor([0.19]),
            terminated=torch.zeros(1),
            cfg=custom,
        )
        assert result.item() is True
        # just above custom thresholds -> failure
        result = compute_step_success_from_errors(
            lin_err_xy=torch.tensor([0.11]),
            ang_err_z=torch.tensor([0.19]),
            terminated=torch.zeros(1),
            cfg=custom,
        )
        assert result.item() is False

    def test_frozen(self):
        with pytest.raises(AttributeError):
            CFG.eps_lin = 0.5


# TaskSampler + TaskIDConfig

class TestTaskIDConfig:
    """Task-ID observation augmentation toggle."""

    def test_default_is_off(self):
        cfg = TaskIDConfig()
        assert cfg.append_task_id is False

    def test_preset_on(self):
        assert TASK_ID_ON.append_task_id is True

    def test_preset_off(self):
        assert TASK_ID_OFF.append_task_id is False

    def test_frozen(self):
        with pytest.raises(AttributeError):
            TASK_ID_ON.append_task_id = False


class TestTaskSampler:
    """Sampling distribution and task-index tests."""

    # uniform distribution

    def test_uniform_default(self):
        """Default sampler uses all 6 training tasks, uniform weights."""
        s = TaskSampler(seed=42)
        assert s.num_tasks == 6
        expected_prob = 1.0 / 6
        for p in s.probabilities.tolist():
            assert p == pytest.approx(expected_prob, abs=1e-6)

    def test_uniform_sampling_hits_all_tasks(self):
        """Over many draws, every task is sampled at least once."""
        s = TaskSampler(seed=0)
        seen: set[str] = set()
        for _ in range(500):
            _, gym_id = s.sample()
            seen.add(gym_id)
        assert seen == set(ALL_TRAIN_TASK_IDS)

    def test_uniform_batch(self):
        """sample_batch returns the correct number of results."""
        s = TaskSampler(seed=7)
        batch = s.sample_batch(100)
        assert len(batch) == 100
        for idx, gym_id in batch:
            assert 0 <= idx < s.num_tasks
            assert gym_id == s.task_ids[idx]

    def test_uniform_distribution_empirical(self):
        """Empirical frequencies should be roughly uniform (chi-squared-ish)."""
        s = TaskSampler(seed=123)
        counts = torch.zeros(s.num_tasks)
        n = 6000
        for _ in range(n):
            idx, _ = s.sample()
            counts[idx] += 1
        freqs = counts / n
        expected = 1.0 / s.num_tasks
        # each freq within 5% of expected
        for f in freqs.tolist():
            assert abs(f - expected) < 0.05, f"freq {f:.3f} too far from {expected:.3f}"

    # weighted distribution

    def test_weighted_deterministic(self):
        """Weight = (1, 0, 0, ...) → always picks task 0."""
        weights = (1.0,) + (0.0,) * 5
        s = TaskSampler(weights=weights, seed=99)
        for _ in range(50):
            idx, gym_id = s.sample()
            assert idx == 0
            assert gym_id == ALL_TRAIN_TASK_IDS[0]

    def test_weighted_two_tasks(self):
        """Weight concentrated on tasks 0 and 5 — others never sampled."""
        weights = (5.0, 0.0, 0.0, 0.0, 0.0, 5.0)
        s = TaskSampler(weights=weights, seed=42)
        seen_indices: set[int] = set()
        for _ in range(200):
            idx, _ = s.sample()
            seen_indices.add(idx)
        assert seen_indices == {0, 5}

    def test_weighted_normalisation(self):
        """Unnormalised weights get normalised to probabilities."""
        s = TaskSampler(
            task_ids=("A", "B", "C"),
            weights=(2.0, 3.0, 5.0),
        )
        probs = s.probabilities.tolist()
        assert probs[0] == pytest.approx(0.2)
        assert probs[1] == pytest.approx(0.3)
        assert probs[2] == pytest.approx(0.5)

    # seed reproducibility

    def test_seed_reproducibility(self):
        """Same seed → same sequence."""
        s1 = TaskSampler(seed=77)
        s2 = TaskSampler(seed=77)
        seq1 = [s1.sample() for _ in range(20)]
        seq2 = [s2.sample() for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_differ(self):
        """Different seeds → (very likely) different sequences."""
        s1 = TaskSampler(seed=1)
        s2 = TaskSampler(seed=2)
        seq1 = [s1.sample()[0] for _ in range(50)]
        seq2 = [s2.sample()[0] for _ in range(50)]
        assert seq1 != seq2

    # task_index lookup

    def test_task_index_valid(self):
        s = TaskSampler()
        for i, tid in enumerate(ALL_TRAIN_TASK_IDS):
            assert s.task_index(tid) == i

    def test_task_index_invalid(self):
        s = TaskSampler()
        with pytest.raises(ValueError):
            s.task_index("nonexistent-env-id")

    # custom task list

    def test_custom_task_ids(self):
        s = TaskSampler(task_ids=("X", "Y"), seed=0)
        assert s.num_tasks == 2
        idx, gym_id = s.sample()
        assert gym_id in ("X", "Y")

    # validation

    def test_empty_task_ids_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            TaskSampler(task_ids=())

    def test_weight_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="len"):
            TaskSampler(weights=(1.0, 2.0))  # 6 tasks, 2 weights

    def test_negative_weights_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            TaskSampler(task_ids=("A", "B"), weights=(-1.0, 1.0))

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="all be zero"):
            TaskSampler(task_ids=("A", "B"), weights=(0.0, 0.0))
