#!/usr/bin/env python3
import requests
import json

print("Testing /reset endpoint...")
resp = requests.post('http://localhost:7860/reset', json={'task_id': 'easy_small_clinic'})
print(f"Status: {resp.status_code}")
data = resp.json()
print(f"Response keys: {list(data.keys())}")
print(json.dumps(data, indent=2)[:1000])
