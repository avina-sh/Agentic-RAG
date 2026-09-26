import json
from utils import load_golden_set, filter_by_type
from workflow import ask_agent
from cache import CACHE_STORE
import time


def timed_call(question):
    start = time.perf_counter()
    result = ask_agent(question)
    return time.perf_counter() - start


def main():
    golden_set = load_golden_set()
    kb_cases = filter_by_type(golden_set, "kb")[:5]  

    CACHE_STORE.clear() 

    rows = []
    for case in kb_cases:
        q = case["question"]

        miss_latency = timed_call(q)                          
        hit_latency_exact = timed_call(q)                     
        hit_latency_rephrase = timed_call(f"Could you explain: {q}")  

        rows.append({
            "question": q,
            "cache_miss_s": round(miss_latency, 3),
            "exact_hit_s": round(hit_latency_exact, 3),
            "semantic_hit_s": round(hit_latency_rephrase, 3),
        })
        print(rows[-1])

    with open("evals/results_cache_benchmark.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()