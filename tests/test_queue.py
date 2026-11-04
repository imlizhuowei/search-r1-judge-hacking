"""Runs scripts/queue_next.sh for real (bash) with a fake on_boot that just records what it was asked to do."""

import os
import shutil
import subprocess

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")


def run_queue(tmp_path, queue_lines, QUEUE="", fail_job=None, extra_env=None):
    log = tmp_path / "calls.txt"
    fake = tmp_path / "fake_on_boot.sh"
    fake.write_text(
        f'echo "$1 EXP=${{EXP:-}} ADV=${{ADV:-}} AFTER=$AFTER" >> {log}\n'
        + (f'[ "$1" = "{fail_job}" ] && exit 3\n' if fail_job else "")
        + "exit 0\n")
    qf = tmp_path / "queue.txt"
    qf.write_text("\n".join(queue_lines) + "\n")
    env = dict(os.environ, REPO=os.path.abspath(REPO), QUEUE_FILE=str(qf), ON_BOOT=f"bash {fake}",
               POLL="0", SKIP_WAIT_CLEANUP="1", EXIT_WHEN_EMPTY="1", QUEUE=QUEUE, BARK_KEY="",
               **(extra_env or {}))
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


def test_failure_skips_rest_of_initial_queue_but_keeps_reading_file(tmp_path):
    r, calls, left = run_queue(tmp_path, ["EXP=fixed smoke_gca2"], QUEUE="smoke_gca train", fail_job="smoke_gca")
    assert r.returncode == 3                                   # last failure code is reported
    assert calls[0] == "smoke_gca EXP= ADV= AFTER=keep"
    assert "train EXP= ADV= AFTER=keep" not in calls           # rest of QUEUE skipped
    assert calls[1] == "smoke_gca2 EXP=fixed ADV= AFTER=keep"  # a job appended after the fix still runs


def test_idle_timeout_shuts_down(tmp_path):
    # IDLE_HOURS=0 -> immediately idle-timeout; replace shutdown with a marker via a fake /usr/bin path is not
    # possible, so run the queue with the shutdown branch turned into "touch marker" by a sed'd copy.
    src = open(os.path.join(REPO, "scripts", "queue_next.sh")).read()
    marker = tmp_path / "shutdown_called"
    patched = tmp_path / "queue_next.sh"
    patched.write_text(src.replace("/usr/bin/shutdown", f"touch {marker}").replace("sleep 60", "true"))
    qf = tmp_path / "queue.txt"
    qf.write_text("")
    env = dict(os.environ, REPO=os.path.abspath(REPO), QUEUE_FILE=str(qf), ON_BOOT="true", POLL="0",
               SKIP_WAIT_CLEANUP="1", IDLE_HOURS="0", QUEUE="", BARK_KEY="")
    r = subprocess.run(["bash", str(patched)], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and marker.exists(), r.stdout + r.stderr
