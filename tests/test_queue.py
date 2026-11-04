"""Runs scripts/queue_next.sh for real (bash) with a fake on_boot that just records what it was asked to do."""

import os
import shutil
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")


def run_queue(tmp_path, queue_lines, QUEUE="", fail_job=None):
    log = tmp_path / "calls.txt"
    fake = tmp_path / "fake_on_boot.sh"
    fake.write_text(
        f'echo "$1 EXP=${{EXP:-}} ADV=${{ADV:-}} AFTER=$AFTER" >> {log}\n'
        + (f'[ "$1" = "{fail_job}" ] && exit 3\n' if fail_job else "")
        + "exit 0\n")
    qf = tmp_path / "queue.txt"
    qf.write_text("\n".join(queue_lines) + "\n")
    env = dict(os.environ, REPO=os.path.abspath(REPO), QUEUE_FILE=str(qf), ON_BOOT=f"bash {fake}",
               POLL="0", SKIP_WAIT_CLEANUP="1", EXIT_WHEN_EMPTY="1", QUEUE=QUEUE, BARK_KEY="")
    r = subprocess.run(["bash", os.path.join(REPO, "scripts", "queue_next.sh")], env=env,
                       capture_output=True, text=True, timeout=30)
    calls = log.read_text().splitlines() if log.exists() else []
    return r, calls, qf.read_text()


@pytest.mark.skipif(shutil.which("bash") is None or shutil.which("python3") is None, reason="needs bash+python3")
def test_queue_runs_initial_then_file_jobs(tmp_path):
    r, calls, left = run_queue(tmp_path, ["# comment", "EXP=judge_s1 ADV=grpo train", "", "EXP=x none"],
                               QUEUE="smoke_gca")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["smoke_gca EXP= ADV= AFTER=keep",
                     "train EXP=judge_s1 ADV=grpo AFTER=keep",
                     "none EXP=x ADV= AFTER=keep"]
    assert "train" not in left and "none" not in left      # consumed lines are removed
    assert "# comment" in left


def test_queue_stops_on_failure(tmp_path):
    r, calls, left = run_queue(tmp_path, ["EXP=a train", "EXP=b train"], QUEUE="smoke_gca", fail_job="smoke_gca")
    assert r.returncode == 3
    assert calls == ["smoke_gca EXP= ADV= AFTER=keep"]      # later jobs not started
    assert "EXP=a train" in left
