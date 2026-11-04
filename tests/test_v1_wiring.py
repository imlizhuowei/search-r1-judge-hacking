"""GCA on the V1 trainer: labels travel reward -> TransferQueue extra_fields -> non_tensor_batch -> advantage."""

import os
import shutil
import subprocess

import numpy as np
import pytest

from gca.reward import compute_score
from gca.verl_plugin import GCA_KEYS, gca_fields_from_extra, gca_numpy

REPO = os.path.join(os.path.dirname(__file__), "..")
TOOL = ("<tool_call>{}</tool_call><tool_response>Doc 1: Thomas Jefferson wrote the Declaration."
        "</tool_response>")


def extra_from_reward(sol, gold):
    """What the V1 agent loop stores per sample: extra_fields["reward_extra_info"] = reward dict minus score."""
    r = compute_score("searchR1_nq", sol, {"target": [gold]}, {"question": "q"})
    return {"reward_extra_info": {k: v for k, v in r.items() if k != "score"}, "turns": 3}, r["score"]


def test_extract_labels_from_extra_fields():
    e1, s1 = extra_from_reward(TOOL + "<answer>Thomas Jefferson</answer>", "Thomas Jefferson")
    e2, s2 = extra_from_reward("<answer>John Adams</answer>", "Thomas Jefferson")
    out = gca_fields_from_extra(np.array([e1, e2], dtype=object))
    assert set(out) == set(GCA_KEYS)
    assert all(v.dtype == object and len(v) == 2 for v in out.values())
    assert (s1, s2) == (1.0, 0.0)
    assert out["gca_first_hit"][1] == -1 and out["gca_answered"].tolist() == [1, 1]


def test_list_input_and_singleton_unwrap():
    out = gca_fields_from_extra([{"reward_extra_info": {"gca_type": ["A"], "gca_first_hit": [0], "gca_answered": [1]}}])
    assert out["gca_type"].tolist() == ["A"] and out["gca_first_hit"].tolist() == [0]


def test_missing_label_fails_loudly():
    # plain EM reward (no gca/reward.py) must not silently train GRPO under the GCA name
    with pytest.raises(KeyError, match="gca_type"):
        gca_fields_from_extra([{"reward_extra_info": {"acc": 1.0}}])
    with pytest.raises(KeyError):
        gca_fields_from_extra([None])


def test_extracted_labels_feed_the_advantage():
    sols = [TOOL + "<answer>Thomas Jefferson</answer>", "<answer>John Adams</answer>"] * 2
    pairs = [extra_from_reward(s, "Thomas Jefferson") for s in sols]
    nt = gca_fields_from_extra([p[0] for p in pairs])
    scores = np.array([p[1] for p in pairs])
    mask = np.ones((4, 12))
    adv, loss_mask = gca_numpy(scores, mask, ["u", "u", "u", "u"], nt["gca_first_hit"], nt["gca_answered"],
                               nt["gca_type"])
    assert adv.shape == (4, 12) and (adv[0] > 0).any() and (adv[1] <= 0).all()


def test_run_scripts_use_v1_trainer():
    for f in ("train.sh", "smoke_train.sh"):
        src = open(os.path.join(REPO, "scripts", f)).read()
        assert "use_v1=False" not in src, f   # baseline and GCA must share the trainer code path


VERL = os.environ.get("VERL_SRC", "/home/claude/refs/verl")


@pytest.mark.skipif(not os.path.isdir(os.path.join(VERL, ".git")) or shutil.which("git") is None,
                    reason="needs a verl checkout")
def test_patches_apply_cleanly_on_pinned_verl():
    for p in ("verl_gca.patch", "verl_gca_v1.patch"):
        path = os.path.abspath(os.path.join(REPO, "patches", p))
        fwd = subprocess.run(["git", "apply", "--check", path], cwd=VERL, capture_output=True, text=True)
        rev = subprocess.run(["git", "apply", "--reverse", "--check", path], cwd=VERL, capture_output=True, text=True)
        assert fwd.returncode == 0 or rev.returncode == 0, p + fwd.stderr
