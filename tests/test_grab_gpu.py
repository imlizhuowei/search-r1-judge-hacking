import importlib.util
import os

import pytest

spec = importlib.util.spec_from_file_location(
    "grab_gpu", os.path.join(os.path.dirname(__file__), "..", "tools", "grab_gpu.py"))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


class FakeResp:
    def __init__(self, j, status=200):
        self._j, self.status_code = j, status

    def json(self):
        return self._j

    def raise_for_status(self):
        pass


def make_api(responses):
    api = g.AutoDL("tok")
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(url)
        return responses.pop(0)

    api.s.post = post
    api.calls = calls
    return api


def test_retries_on_no_capacity_then_succeeds(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    api = make_api([FakeResp({"code": "Fail", "msg": "暂无空闲机器"})] * 3 + [FakeResp({"code": "Success"})])
    g.grab(api, "u", interval=3, bark=None, max_hours=1)
    assert len(api.calls) == 4


def test_auth_failure_stops(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    api = make_api([FakeResp({"code": "AuthorizeFailed", "msg": ""})])
    with pytest.raises(g.Fatal):
        g.grab(api, "u", interval=3, bark=None, max_hours=1)


def test_unknown_refusal_stops_instead_of_looping(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    api = make_api([FakeResp({"code": "Weird", "msg": "实例已释放"})])
    with pytest.raises(g.Fatal):
        g.grab(api, "u", interval=3, bark=None, max_hours=1)


def test_interval_floor(monkeypatch):
    slept = []
    monkeypatch.setattr(g.time, "sleep", lambda s: slept.append(s))
    api = make_api([FakeResp({"code": "Fail", "msg": "无空闲GPU"}), FakeResp({"code": "Success"})])
    g.grab(api, "u", interval=0.1, bark=None, max_hours=1)
    assert min(slept) >= g.MIN_INTERVAL * 0.8


def test_status_lookup():
    api = make_api([FakeResp({"code": "Success", "data": {"list": [{"uuid": "a", "status": "shutdown"},
                                                                    {"uuid": "u", "status": "running"}],
                                                           "result_total": 2}})])
    assert api.status("u") == "running"


def test_rate_limit_backs_off_then_continues(monkeypatch):
    slept = []
    monkeypatch.setattr(g.time, "sleep", lambda s: slept.append(s))
    api = make_api([FakeResp({}, status=429), FakeResp({"code": "Success"})])
    g.grab(api, "u", interval=1, bark=None, max_hours=1)
    assert 5 in slept and len(api.calls) == 2


def test_status_survives_network_drop(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    api = g.AutoDL("tok")
    seq = [ConnectionError("hotspot down"), ConnectionError("still down"),
           FakeResp({"code": "Success", "data": {"list": [{"uuid": "u", "status": "running"}], "result_total": 1}})]

    def post(url, json=None, timeout=None):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x

    api.s.post = post
    assert api.status("u") == "running"


def test_launch_ok_even_if_ssh_hangs(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)

    def fake_run(cmd, **kw):
        assert "-n" in cmd and "LAUNCHED" in cmd[-1]
        raise g.subprocess.TimeoutExpired(cmd, 60, output="LAUNCHED\n")

    monkeypatch.setattr(g.subprocess, "run", fake_run)
    g.launch("ssh -p 1 root@h", "smoke", None)  # must not raise


def test_launch_retries_then_fails(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    calls = []

    class R:
        stdout, stderr, returncode = "", "Connection refused", 255

    monkeypatch.setattr(g.subprocess, "run", lambda cmd, **kw: calls.append(1) or R())
    with pytest.raises(g.Fatal):
        g.launch("ssh -p 1 root@h", "smoke", None)
    assert len(calls) == 30


def test_launch_missing_on_boot_schedules_shutdown(monkeypatch):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)

    class R:
        stdout, stderr, returncode = "MISSING_ON_BOOT\n", "", 0

    monkeypatch.setattr(g.subprocess, "run", lambda cmd, **kw: R())
    with pytest.raises(g.Fatal):
        g.launch("ssh -p 1 root@h", "smoke", None)


def _capture_remote(monkeypatch, tmp_path):
    """Run launch() with a fake ssh that executes the remote command in bash, against a fake repo."""
    import subprocess as sp
    real_run = sp.run  # keep the real one before monkeypatching (g.subprocess is the same module)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    monkeypatch.setattr(g, "REMOTE_REPO", str(repo))
    monkeypatch.setattr(g, "REMOTE_LOG", str(tmp_path / "on_boot.log"))
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    ran = {}

    def fake_run(cmd, **kw):
        ran["cmd"] = cmd
        remote = cmd[-1].replace("git pull -q", "true").replace("setsid nohup ", "")
        return real_run(["bash", "-c", remote], capture_output=True, text=True, timeout=20)

    monkeypatch.setattr(g.subprocess, "run", fake_run)
    return repo, ran


def test_remote_command_really_runs_in_bash(monkeypatch, tmp_path):
    repo, ran = _capture_remote(monkeypatch, tmp_path)
    out = tmp_path / "env.txt"
    (repo / "scripts" / "on_boot.sh").write_text(f'echo "$1 $AFTER $MAX_RUN_HOURS $BARK_KEY" > {out}\n')
    g.launch("ssh -p 40606 root@h", "smoke", "KEY", after="shutdown", max_run_hours=3)
    import time as _t
    for _ in range(50):
        if out.exists() and out.read_text().strip():
            break
        _t.sleep(0.05)
    assert out.read_text().split() == ["smoke", "shutdown", "3", "KEY"]   # env vars reached on_boot.sh
    cmd = ran["cmd"]
    assert cmd[0] == "ssh" and cmd.index("-n") < cmd.index("root@h")      # options before destination


def test_remote_command_missing_on_boot(monkeypatch, tmp_path):
    _capture_remote(monkeypatch, tmp_path)
    # no on_boot.sh in the fake repo -> MISSING branch; neutralise the scheduled shutdown
    fake = g.subprocess.run
    monkeypatch.setattr(g.subprocess, "run",
                        lambda cmd, **kw: fake(cmd[:-1] + [cmd[-1].replace("/usr/bin/shutdown", "true")], **kw))
    with pytest.raises(g.Fatal):
        g.launch("ssh -p 1 root@h", "smoke", None)


def test_main_powers_off_if_launch_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    tok = tmp_path / "t"; tok.write_text("tok")
    calls = []
    statuses = iter(["shutdown", "running"])
    monkeypatch.setattr(g.AutoDL, "status", lambda self, u, retries=60: next(statuses))
    monkeypatch.setattr(g.AutoDL, "power_on", lambda self, u: calls.append("on"))
    monkeypatch.setattr(g.AutoDL, "power_off", lambda self, u: calls.append("off"))

    def bad_launch(*a, **k):
        raise g.Fatal("ssh failed: port changed")

    monkeypatch.setattr(g, "launch", bad_launch)
    monkeypatch.setattr(g.sys, "argv", ["x", "--instance", "u", "--token-file", str(tok), "--ssh", "ssh -p 1 root@h"])
    with pytest.raises(SystemExit):
        g.main()
    assert calls == ["on", "off"]


def test_main_no_power_off_if_never_grabbed(monkeypatch, tmp_path):
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    tok = tmp_path / "t"; tok.write_text("tok")
    calls = []
    monkeypatch.setattr(g.AutoDL, "status", lambda self, u, retries=60: "shutdown")
    monkeypatch.setattr(g.AutoDL, "power_off", lambda self, u: calls.append("off"))

    def no_card(*a, **k):
        raise g.Fatal("token expired")

    monkeypatch.setattr(g, "grab", no_card)
    monkeypatch.setattr(g.sys, "argv", ["x", "--instance", "u", "--token-file", str(tok)])
    with pytest.raises(SystemExit):
        g.main()
    assert calls == []
