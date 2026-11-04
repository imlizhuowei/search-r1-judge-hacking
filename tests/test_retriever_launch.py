"""retriever/launch.sh must never run two retrievers (each holds a ~60 GB index in RAM)."""

import os
import shutil
import subprocess
import time

import pytest

REPO = os.path.join(os.path.dirname(__file__), "..")


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock")
def test_second_launch_refuses(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python").write_text(f"#!/bin/sh\necho started >> {tmp_path}/starts\nsleep 30\n")
    (fake / "python").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}:{os.environ['PATH']}", LOCK=str(tmp_path / "lock"))
    script = os.path.join(REPO, "retriever", "launch.sh")
    first = subprocess.Popen(["bash", script], env=env, start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        second = subprocess.run(["bash", script], env=env, capture_output=True, text=True, timeout=10)
        assert second.returncode == 0 and "not starting a second one" in second.stderr
        assert (tmp_path / "starts").read_text().count("started") == 1
    finally:
        os.killpg(first.pid, 9)
