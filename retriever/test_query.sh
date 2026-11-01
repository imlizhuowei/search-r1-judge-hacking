#!/usr/bin/env bash
# Send one test query to the running retrieval server.
curl -s -X POST "http://127.0.0.1:${PORT:-8000}/retrieve" \
  -H 'Content-Type: application/json' \
  -d '{"queries": ["Who wrote the Declaration of Independence?"], "topk": 3, "return_scores": true}' | python -m json.tool | head -40
