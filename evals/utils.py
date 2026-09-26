import json
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import ask_agent

def load_golden_set(path="evals/golden_set.json"):
    with open(path) as f:
        return json.load(f)

def run_agent_timed(question:str):
    start=time.perf_counter()
    result=ask_agent(question)
    elapsed=time.perf_counter()-start
    return result,elapsed

def filter_by_type(golden_set,type_name):
    return [case for case in golden_set if case["type"]==type_name]

