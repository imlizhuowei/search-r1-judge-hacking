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
