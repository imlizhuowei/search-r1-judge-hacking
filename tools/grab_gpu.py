"""Keep trying to power on an AutoDL instance with GPU; when it starts, launch our job over SSH and notify.

Runs on your own computer (Windows/macOS/Linux) with Python 3.9+ and `pip install requests`.
The computer must stay awake while it runs.

Uses the AutoDL *console* endpoints (same as the web page), authenticated with your console login token.
That is not a public API: the token is as powerful as your password, and automated use may break the
platform's terms. Keep the token in a file only you can read, never in git or chat.

Setup (once):
  1. Console token: log in to autodl.com, F12 -> Network -> click any request to www.autodl.com/api/v1/...
     -> Request Headers -> copy the value of "Authorization". Save it to ~/.autodl_token (one line).
  2. Instance UUID: shown in the console instance list (e.g. 66714bb469-c3f0be79).
  3. SSH key login (so no password is needed): see tools/README.md.
  4. Phone push (optional): install the Bark app (iPhone), copy your key.

Run:
  python tools/grab_gpu.py --instance 66714bb469-c3f0be79 \
      --ssh "ssh -p 12345 root@connect.xxx.seetacloud.com" --bark-key XXXX
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
import urllib.parse

import requests

BASE = "https://www.autodl.com"
CAPACITY_HINTS = ("无空闲", "没有空闲", "暂无空闲", "资源不足", "库存不足", "GPU不足", "已售罄")
TERMINAL = {"AuthorizeFailed": "登录 token 失效，请重新复制", "TORecharge": "余额不足，请充值",
            "TORealName": "需要实名认证"}
MIN_INTERVAL = 1.0  # seconds; one request at a time, never concurrent


class Capacity(Exception):
    pass


class Fatal(Exception):
    pass


class RateLimited(Exception):
    pass


def notify(bark_key: str | None, title: str, body: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {title} {body}", flush=True)
    if bark_key:
        try:
            url = f"https://api.day.app/{bark_key}/{urllib.parse.quote(title)}/{urllib.parse.quote(body or ' ')}"
            requests.get(url, params={"level": "timeSensitive", "sound": "alarm"}, timeout=10)
        except Exception as e:  # notification must never kill the loop
            print("  (push failed:", e, ")")


class AutoDL:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": token, "Accept": "application/json"})

    def _post(self, path: str, body: dict, power_on: bool = False):
        r = self.s.post(BASE + path, json=body, timeout=20)
        if r.status_code in (401, 403):
            raise Fatal(TERMINAL["AuthorizeFailed"])
        if r.status_code == 429:
            raise RateLimited("429")
        r.raise_for_status()
        j = r.json()
        code, msg = j.get("code"), str(j.get("msg", ""))
        if code == "Success":
            return j.get("data")
        if code in TERMINAL:
            raise Fatal(TERMINAL[code])
        if power_on and any(h in msg.replace(" ", "") for h in CAPACITY_HINTS):
            raise Capacity(msg)
        raise Fatal(f"AutoDL refused: code={code} msg={msg[:120]}")

    def power_on(self, uuid: str) -> None:
        self._post("/api/v1/instance/power_on", {"instance_uuid": uuid}, power_on=True)

    def status(self, uuid: str, retries: int = 60) -> str | None:
        """Retries through network drops (flaky hotspot): up to ~10 min."""
        for i in range(retries):
            try:
                return self._status_once(uuid)
            except (Fatal, Capacity):
                raise
            except Exception as e:
                print(f"  查询状态失败（网络？）{i + 1}/{retries}: {str(e)[:60]}", flush=True)
                time.sleep(10)
        raise Fatal("网络长时间不通，无法确认实例状态")

    def _status_once(self, uuid: str) -> str | None:
        page = 1
        while True:
            data = self._post("/api/v1/instance", {"page_index": page, "page_size": 100, "status": [],
                                                   "charge_type": [], "sub_name": "", "unbind_sub_user": False})
            for row in data.get("list", []):
                if row.get("uuid") == uuid:
                    return row.get("status")
            if page * 100 >= data.get("result_total", 0):
                return None
            page += 1


def grab(api: AutoDL, uuid: str, interval: float, bark: str | None, max_hours: float) -> None:
    interval = max(interval, MIN_INTERVAL)
    t_end = time.time() + max_hours * 3600
    n = 0
    n_429 = 0
    while time.time() < t_end:
        n += 1
        try:
            api.power_on(uuid)
            n_429 = 0
            notify(bark, "抢到卡了", f"第 {n} 次尝试成功，正在开机")
            return
        except Capacity as e:
            if n % 60 == 1:
                print(f"[{time.strftime('%H:%M:%S')}] 无卡，继续等待 (#{n}): {str(e)[:40]}", flush=True)
        except RateLimited:
            backoff = min(60, 5 * 2 ** min(n_429, 4))
            n_429 += 1
            print(f"  被限流，暂停 {backoff}s", flush=True)
            time.sleep(backoff)
            continue
        except Fatal:
            raise
        except Exception as e:  # network hiccup: back off a little
            print("  network error:", e)
            time.sleep(10)
        time.sleep(interval * random.uniform(0.8, 1.2))
    raise Fatal(f"{max_hours} 小时内没抢到卡")


def wait_running(api: AutoDL, uuid: str, timeout_s: int = 900) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        st = api.status(uuid)
        if st == "running":
            return
        if st and st.startswith("shutdown"):
            raise Fatal(f"开机失败，状态 {st}")
        time.sleep(5)
    raise Fatal("开机超时")


def launch(ssh_cmd: str, job: str, bark: str | None, after: str = "keep", max_run_hours: float = 3) -> None:
    """Start on_boot.sh on the server, fully detached.

    AutoDL sessions may stay open after the command finishes (seen with `ssh ... echo ok`), so we do not
    wait for ssh to exit: the remote side prints LAUNCHED once the job is in the background, and we
    stop waiting after 60 s and look for that marker in whatever output arrived."""
    remote = (f"cd /root/autodl-tmp/search-r1-judge-hacking && git pull -q; "
              f"BARK_KEY={bark or ''} AFTER={after} MAX_RUN_HOURS={int(max_run_hours)} "
              f"setsid nohup bash scripts/on_boot.sh {job} > /root/autodl-tmp/on_boot.log 2>&1 < /dev/null & "
              f"echo LAUNCHED")
    cmd = ssh_cmd.split() + ["-n", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
                             "-o", "ConnectTimeout=20", remote]
    err = ""
    for _ in range(30):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out, err = r.stdout or "", r.stderr or ""
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = "ssh session did not close (timeout)"
        if "LAUNCHED" in out:
            notify(bark, "任务已在服务器上启动", f"on_boot.sh {job}")
            return
        time.sleep(20)
    notify(bark, "SSH 启动任务失败，卡在空转！", "请尽快手动登录运行 on_boot.sh 或关机")
    raise Fatal("ssh failed: " + err[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", required=True, help="instance UUID from the console")
    ap.add_argument("--token-file", default=os.path.expanduser("~/.autodl_token"))
    ap.add_argument("--interval", type=float, default=1.0, help=f"seconds between tries (min {MIN_INTERVAL})")
    ap.add_argument("--max-hours", type=float, default=24)
    ap.add_argument("--ssh", help='e.g. "ssh -p 12345 root@connect.xxx.seetacloud.com" (key login)')
    ap.add_argument("--job", default="smoke", help="argument passed to scripts/on_boot.sh")
    ap.add_argument("--bark-key", default=os.environ.get("BARK_KEY"))
    ap.add_argument("--max-run-hours", type=float, default=3,
                    help="server powers itself off after this many hours no matter what (0 = never)")
    ap.add_argument("--after", default="shutdown", choices=["keep", "shutdown", "shutdown_always"],
                    help="when the job ends: keep the GPU / shut down if it succeeded / always shut down")
    a = ap.parse_args()

    token = os.environ.get("AUTODL_TOKEN") or open(a.token_file).read().strip()
    api = AutoDL(token)
    try:
        st = api.status(a.instance)
        print("当前状态:", st)
        if st != "running":
            grab(api, a.instance, a.interval, a.bark_key, a.max_hours)
            wait_running(api, a.instance)
        notify(a.bark_key, "实例已运行", "GPU 模式")
        if a.ssh:
            launch(a.ssh, a.job, a.bark_key, a.after, a.max_run_hours)
    except Fatal as e:
        notify(a.bark_key, "抢卡脚本停止", str(e))
        sys.exit(1)
    except Exception as e:  # unexpected crash: the instance may be running and billing
        notify(a.bark_key, "抢卡脚本崩溃，检查实例是否在空转", str(e)[:80])
        raise


if __name__ == "__main__":
    main()
