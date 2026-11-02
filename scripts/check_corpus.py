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
print("last real line:", json.loads(read(real - 1))["id"], "| first:", json.loads(read(0))["id"])
bad = 0
random.seed(0)
for i in random.sample(range(real), 2000):
    try:
        json.loads(read(i))
    except Exception:
        bad += 1
print("random 2000 lines unparseable:", bad)
