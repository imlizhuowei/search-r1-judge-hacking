"""Sanity-check wiki-18.jsonl against the cached line offsets.

Usage (retriever venv): python scripts/check_corpus.py [/root/autodl-fs/searchr1]
Prints the size of any junk tail, the last real line, and parses random lines as JSON.
"""
import json
import os
import random
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-fs/searchr1"
o = np.load(os.path.join(d, "wiki-18.offsets.npy"))
fd = os.open(os.path.join(d, "wiki-18.jsonl"), os.O_RDONLY)
read = lambda i: os.pread(fd, int(o[i + 1] - o[i]), int(o[i]))
n = len(o) - 1
print("lines in offsets:", n, "| file size:", os.path.getsize(os.path.join(d, "wiki-18.jsonl")))
last = read(n - 1)
print("last line:", len(last), "bytes; all NUL/whitespace:", not last.strip(b" \t\r\n\x00"))
real = n - 1 if not last.strip(b" \t\r\n\x00") else n


def check(i):
    try:
        return json.loads(read(i))["id"] == str(i + 1)
    except Exception:
        return False


print("first line ok:", check(0), "| last real line ok:", check(real - 1))
for k in range(2, 6):
    print(f"line {real - k + 1} ok:", check(real - k))
random.seed(0)
bad = [i for i in random.sample(range(real), 2000) if not check(i)]
print("random 2000 lines bad (unparseable or id != line number):", len(bad), bad[:10])
