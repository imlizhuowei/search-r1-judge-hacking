"""Only runs where torch + verl are installed (the server). Checks our numpy GRPO baseline equals verl's,
and that the registered "gca" estimator with eta=0 reproduces verl's GRPO exactly (exit check 1, in verl)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")

from gca.verl_plugin import grpo_scalar_advantage  # noqa: E402
import gca.verl_plugin  # noqa: E402,F401  registers "gca"


def _fake_batch(seed=0, n_groups=3, n=5, T=12):
    rng = np.random.default_rng(seed)
    bs = n_groups * n
    mask = np.zeros((bs, T), dtype=np.float32)
    for i in range(bs):  # [search][tool][answer][pad]
        mask[i, :3] = 1
        mask[i, 6:9] = 1
    scores = rng.integers(0, 2, size=bs).astype(np.float32)
    rewards = np.zeros((bs, T), dtype=np.float32)
    rewards[:, 8] = scores
    index = np.array([f"g{i // n}" for i in range(bs)], dtype=object)
    types = rng.choice(["A", "B1", "C", "D"], size=bs)
    ntb = {
        "gca_type": types,
        "gca_first_hit": np.where(np.isin(types, ["A", "B1"]), 1, -1),
        "gca_answered": np.ones(bs, dtype=int),
    }
    return torch.tensor(rewards), torch.tensor(mask), index, ntb, scores


def test_numpy_grpo_equals_verl():
    rew, mask, index, _, scores = _fake_batch()
    verl_adv, _ = core_algos.compute_grpo_outcome_advantage(rew, mask, index)
    ours = grpo_scalar_advantage(scores, index)
    np.testing.assert_allclose(verl_adv.numpy(), ours[:, None] * mask.numpy(), atol=1e-6)


def test_registered_gca_eta0_equals_verl_grpo():
    rew, mask, index, ntb, _ = _fake_batch(seed=1)
    fn = core_algos.get_adv_estimator_fn("gca")
    adv, _ = fn(rew, mask, index=index, config={"gca_eta": 0.0}, non_tensor_batch=ntb)
    verl_adv, _ = core_algos.compute_grpo_outcome_advantage(rew, mask, index)
    torch.testing.assert_close(adv, verl_adv, atol=1e-6, rtol=0)
